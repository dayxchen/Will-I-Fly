import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Open-Meteo hourly variable -> model feature name
OPEN_METEO_VARIABLE_MAP = {
    "temperature_2m": "temperature",
    "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation",
    "visibility": "visibility",
}

US_STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}


@dataclass
class WeatherConfig:
    archive_url: str
    geocoding_url: str
    feature_fields: List[str]
    hourly_variables: List[str] = field(
        default_factory=lambda: [
            "temperature_2m",
            "wind_speed_10m",
            "precipitation",
            "visibility",
        ]
    )
    coords_cache_path: Optional[str] = None
    request_delay_seconds: float = 0.1


def build_weather_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unique (date, origin airport, dep hour) combinations for weather queries.
    Expects flight_date, ORIGIN_AIRPORT_ID, dep_hour, and optionally city/state for geocoding.
    """
    cols = ["flight_date", "ORIGIN_AIRPORT_ID", "dep_hour"]
    if "ORIGIN_CITY_NAME" in df.columns:
        cols.append("ORIGIN_CITY_NAME")
    if "ORIGIN_STATE_ABR" in df.columns:
        cols.append("ORIGIN_STATE_ABR")

    keys = df[cols].dropna(subset=["dep_hour"]).copy()
    keys["date_str"] = keys["flight_date"].dt.strftime("%Y-%m-%d")
    keys["dep_hour"] = keys["dep_hour"].astype(int)
    dedupe_cols = ["date_str", "ORIGIN_AIRPORT_ID", "dep_hour"]
    keys = keys.drop_duplicates(subset=dedupe_cols)
    return keys


def _parse_city_name(city_name: str) -> str:
    return str(city_name).strip().strip('"').split(",")[0].strip()


def _pick_geocode_result(results: List[Dict], state_abr: str) -> Optional[Dict]:
    if not results:
        return None

    state_name = US_STATE_ABBR_TO_NAME.get(state_abr.upper())
    if state_name:
        for result in results:
            admin1 = result.get("admin1")
            if admin1 and state_name.lower() in admin1.lower():
                return result

    return results[0]


def geocode_airport(
    city_name: str,
    state_abr: str,
    session: requests.Session,
    config: WeatherConfig,
) -> Optional[Tuple[float, float, str]]:
    """Resolve airport coordinates and timezone via Open-Meteo geocoding."""
    city = _parse_city_name(city_name)
    params = {
        "name": city,
        "count": 10,
        "language": "en",
        "format": "json",
        "country_code": "US",
    }

    resp = session.get(config.geocoding_url, params=params, timeout=15)
    if resp.status_code != 200:
        return None

    results = resp.json().get("results") or []
    match = _pick_geocode_result(results, state_abr)
    if not match:
        return None

    return float(match["latitude"]), float(match["longitude"]), match.get("timezone", "UTC")


def load_coords_cache(path: str) -> Dict[int, Dict]:
    if not os.path.exists(path):
        return {}

    cache_df = pd.read_csv(path)
    records: Dict[int, Dict] = {}
    for _, row in cache_df.iterrows():
        records[int(row["ORIGIN_AIRPORT_ID"])] = row.to_dict()
    return records


def save_coords_cache(path: str, records: Dict[int, Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(list(records.values())).to_csv(path, index=False)


def resolve_airport_coords(
    airport_id: int,
    city_name: str,
    state_abr: str,
    session: requests.Session,
    config: WeatherConfig,
    coords_cache: Dict[int, Dict],
) -> Optional[Tuple[float, float, str]]:
    if airport_id in coords_cache:
        row = coords_cache[airport_id]
        return float(row["latitude"]), float(row["longitude"]), row.get("timezone", "UTC")

    resolved = geocode_airport(city_name, state_abr, session, config)
    if resolved is None:
        return None

    lat, lon, timezone = resolved
    coords_cache[airport_id] = {
        "ORIGIN_AIRPORT_ID": airport_id,
        "ORIGIN_CITY_NAME": city_name,
        "ORIGIN_STATE_ABR": state_abr,
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
    }
    return lat, lon, timezone


def fetch_weather_for_airport_date(
    latitude: float,
    longitude: float,
    timezone: str,
    date_str: str,
    hours_needed: List[int],
    session: requests.Session,
    config: WeatherConfig,
) -> Dict[int, Dict[str, float]]:
    """Fetch hourly weather for one airport on one date; return map hour -> features."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(config.hourly_variables),
        "timezone": timezone,
    }

    resp = session.get(config.archive_url, params=params, timeout=30)
    if resp.status_code != 200:
        return {}

    payload = resp.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {}

    by_hour: Dict[int, Dict[str, float]] = {}
    needed = set(hours_needed)

    for idx, time_str in enumerate(times):
        try:
            hour = int(time_str.split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            continue

        if hour not in needed:
            continue

        features: Dict[str, float] = {}
        complete = True
        for om_var in config.hourly_variables:
            values = hourly.get(om_var)
            if not values or idx >= len(values):
                complete = False
                break
            value = values[idx]
            feature_name = OPEN_METEO_VARIABLE_MAP.get(om_var, om_var)
            if value is None:
                if feature_name == "visibility":
                    value = 24140.0
                else:
                    complete = False
                    break
            features[feature_name] = float(value)

        if complete:
            by_hour[hour] = features

    return by_hour


def build_weather_table(
    keys: pd.DataFrame,
    config: WeatherConfig,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch weather from Open-Meteo and build a table keyed by date, airport, and hour.
    Uses CSV caches for weather and airport coordinates.
    """
    cache_records: Dict[Tuple[str, int, int], Dict] = {}

    if cache_path and os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        for _, row in cache_df.iterrows():
            key = (row["date_str"], int(row["ORIGIN_AIRPORT_ID"]), int(row["dep_hour"]))
            cache_records[key] = row.to_dict()

    coords_cache: Dict[int, Dict] = {}
    if config.coords_cache_path:
        coords_cache = load_coords_cache(config.coords_cache_path)

    session = requests.Session()
    groups = keys.groupby(["ORIGIN_AIRPORT_ID", "date_str"], sort=False)

    for (airport_id, date_str), group in tqdm(groups, desc="Fetching weather"):
        airport_id = int(airport_id)
        hours_needed = group["dep_hour"].astype(int).tolist()
        missing_hours = [
            hour
            for hour in hours_needed
            if (date_str, airport_id, hour) not in cache_records
        ]
        if not missing_hours:
            continue

        if "ORIGIN_CITY_NAME" not in group.columns or "ORIGIN_STATE_ABR" not in group.columns:
            continue

        city_name = group.iloc[0]["ORIGIN_CITY_NAME"]
        state_abr = group.iloc[0]["ORIGIN_STATE_ABR"]
        coords = resolve_airport_coords(
            airport_id, city_name, state_abr, session, config, coords_cache
        )
        if coords is None:
            continue

        lat, lon, timezone = coords
        by_hour = fetch_weather_for_airport_date(
            lat, lon, timezone, date_str, missing_hours, session, config
        )

        for hour in missing_hours:
            features = by_hour.get(hour)
            if not features:
                continue
            record = {
                "date_str": date_str,
                "ORIGIN_AIRPORT_ID": airport_id,
                "dep_hour": hour,
                **features,
            }
            cache_records[(date_str, airport_id, hour)] = record

        if config.request_delay_seconds > 0:
            time.sleep(config.request_delay_seconds)

    if config.coords_cache_path and coords_cache:
        save_coords_cache(config.coords_cache_path, coords_cache)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        weather_df = pd.DataFrame(list(cache_records.values()))
        weather_df.to_csv(cache_path, index=False)
    else:
        weather_df = pd.DataFrame(list(cache_records.values()))

    return weather_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache weather data for flights.")
    parser.add_argument("--flights_parquet", required=True, help="Prepared flights parquet from data_prep.py")
    parser.add_argument(
        "--archive_url",
        default="https://historical-forecast-api.open-meteo.com/v1/forecast",
        help="Open-Meteo weather API URL (historical forecast recommended for visibility)",
    )
    parser.add_argument(
        "--geocoding_url",
        default="https://geocoding-api.open-meteo.com/v1/search",
        help="Open-Meteo geocoding API URL",
    )
    parser.add_argument(
        "--feature_fields",
        default="temperature,wind_speed,precipitation,visibility",
        help="Comma-separated weather feature names to store",
    )
    parser.add_argument(
        "--cache_csv",
        required=True,
        help="Path to a CSV file used as a persistent weather cache.",
    )
    parser.add_argument(
        "--coords_cache_csv",
        default="data/airport_coords_cache.csv",
        help="Path to cache airport geocoding results.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.flights_parquet)
    keys = build_weather_keys(df)

    feature_fields = [f.strip() for f in args.feature_fields.split(",") if f.strip()]
    config = WeatherConfig(
        archive_url=args.archive_url,
        geocoding_url=args.geocoding_url,
        feature_fields=feature_fields,
        coords_cache_path=args.coords_cache_csv,
    )

    build_weather_table(keys, config, cache_path=args.cache_csv)


if __name__ == "__main__":
    main()
