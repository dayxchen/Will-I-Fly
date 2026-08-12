"""Geocode airports by IATA code and map to model-friendly identifiers."""
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from weather_fetch import US_STATE_ABBR_TO_NAME

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_COORDS_CACHE = "data/airport_iata_coords.csv"
AIRPORTS_CSV = Path(__file__).resolve().parent / "airports_iata.csv"

_state_name_to_abbr = {name.lower(): abbr for abbr, name in US_STATE_ABBR_TO_NAME.items()}


def iata_to_airport_id(iata: str) -> int:
    """Stable numeric ID from an IATA airport code."""
    value = 0
    for char in iata.upper():
        value = value * 37 + ord(char)
    return 10000 + (value % 990000)


def _admin1_to_state_abbr(admin1: Optional[str], country_code: Optional[str]) -> str:
    if not admin1:
        return country_code or "XX"
    if country_code == "US":
        for name, abbr in _state_name_to_abbr.items():
            if name in admin1.lower():
                return abbr
    return admin1[:2].upper()


def _load_static_airports() -> Dict[str, Dict[str, str]]:
    by_iata: Dict[str, Dict[str, str]] = {}
    by_icao: Dict[str, Dict[str, str]] = {}
    if not AIRPORTS_CSV.exists():
        return {"by_iata": by_iata, "by_icao": by_icao}

    with open(AIRPORTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            record = {
                "iata": row["iata"].upper(),
                "icao": row["icao"].upper(),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "name": row["name"],
                "iso_country": row["iso_country"],
            }
            by_iata[record["iata"]] = record
            if record["icao"]:
                by_icao[record["icao"]] = record

    return {"by_iata": by_iata, "by_icao": by_icao}


class AirportGeoCache:
    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or DEFAULT_COORDS_CACHE
        self._cache: Dict[str, Dict] = {}
        static = _load_static_airports()
        self._static_by_iata = static["by_iata"]
        self._static_by_icao = static["by_icao"]
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        df = pd.read_csv(self.cache_path)
        for _, row in df.iterrows():
            self._cache[str(row["iata"]).upper()] = row.to_dict()

    def save(self) -> None:
        if not self._cache:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            pd.DataFrame(list(self._cache.values())).to_csv(self.cache_path, index=False)
        except OSError:
            # Filesystem is read-only in this environment (e.g. Vercel serverless) — skip caching
            pass

    def get(self, iata: str) -> Optional[Dict]:
        return self._cache.get(iata.upper())

    def iata_from_icao(self, icao: Optional[str]) -> Optional[str]:
        if not icao:
            return None
        static = self._static_by_icao.get(icao.upper())
        return static["iata"] if static else None

    def _record_from_static(
        self,
        static: Dict[str, str],
        airport_name: str,
        timezone: Optional[str],
    ) -> Dict:
        country_code = static["iso_country"]
        state_abr = country_code if country_code != "US" else "US"
        return {
            "iata": static["iata"],
            "airport_name": airport_name or static["name"],
            "latitude": float(static["latitude"]),
            "longitude": float(static["longitude"]),
            "timezone": timezone or "UTC",
            "country_code": country_code,
            "state_abr": state_abr,
            "airport_id": iata_to_airport_id(static["iata"]),
        }

    def _geocode_search(
        self,
        session: requests.Session,
        queries: List[str],
    ) -> Optional[Dict]:
        seen = set()
        for query in queries:
            cleaned = query.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)

            params = {
                "name": cleaned,
                "count": 10,
                "language": "en",
                "format": "json",
            }
            resp = session.get(GEOCODING_URL, params=params, timeout=15)
            if resp.status_code != 200:
                continue

            results = resp.json().get("results") or []
            if results:
                return results[0]
        return None

    def resolve(
        self,
        iata: str,
        airport_name: str,
        session: requests.Session,
        icao: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict]:
        iata = iata.upper()
        cached = self.get(iata)
        if cached:
            return cached

        static = self._static_by_iata.get(iata)
        if not static and icao:
            static = self._static_by_icao.get(icao.upper())
        if static:
            record = self._record_from_static(static, airport_name, timezone)
            self._cache[iata] = record
            return record

        search_name = airport_name.split(",")[0].strip()
        queries = [
            airport_name,
            search_name,
            f"{search_name} airport",
            f"{iata} airport",
        ]
        if icao:
            queries.append(f"{icao.upper()} airport")

        match = self._geocode_search(session, queries)
        if not match:
            return None

        country_code = match.get("country_code")
        state_abr = _admin1_to_state_abbr(match.get("admin1"), country_code)
        record = {
            "iata": iata,
            "airport_name": airport_name,
            "latitude": float(match["latitude"]),
            "longitude": float(match["longitude"]),
            "timezone": timezone or match.get("timezone", "UTC"),
            "country_code": country_code,
            "state_abr": state_abr,
            "airport_id": iata_to_airport_id(iata),
        }
        self._cache[iata] = record
        return record


def fetch_current_weather(
    latitude: float,
    longitude: float,
    session: requests.Session,
    timezone: Optional[str] = None,
) -> Dict[str, float]:
    """Fetch current weather at an airport using Open-Meteo forecast API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m,precipitation,visibility",
        "wind_speed_unit": "kmh",
        "timezone": timezone or "auto",
    }
    resp = session.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    resp.raise_for_status()
    current = resp.json().get("current") or {}

    visibility = current.get("visibility")
    if visibility is None:
        visibility = 24140.0

    return {
        "temperature": float(current.get("temperature_2m", 20.0)),
        "wind_speed": float(current.get("wind_speed_10m", 5.0)),
        "precipitation": float(current.get("precipitation", 0.0)),
        "visibility": float(visibility),
    }
