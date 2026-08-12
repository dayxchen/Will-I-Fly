"""Geocode airports by IATA code and map to model-friendly identifiers."""
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

from weather_fetch import US_STATE_ABBR_TO_NAME

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_COORDS_CACHE = "data/airport_iata_coords.csv"

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


class AirportGeoCache:
    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or DEFAULT_COORDS_CACHE
        self._cache: Dict[str, Dict] = {}
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

    def resolve(
        self,
        iata: str,
        airport_name: str,
        session: requests.Session,
    ) -> Optional[Dict]:
        iata = iata.upper()
        cached = self.get(iata)
        if cached:
            return cached

        search_name = airport_name.split(",")[0].strip()
        if "airport" not in search_name.lower():
            search_name = f"{search_name} airport"

        params = {
            "name": search_name,
            "count": 10,
            "language": "en",
            "format": "json",
        }
        resp = session.get(GEOCODING_URL, params=params, timeout=15)
        if resp.status_code != 200:
            return None

        results = resp.json().get("results") or []
        match = results[0] if results else None
        if not match:
            return None

        country_code = match.get("country_code")
        state_abr = _admin1_to_state_abbr(match.get("admin1"), country_code)
        record = {
            "iata": iata,
            "airport_name": airport_name,
            "latitude": float(match["latitude"]),
            "longitude": float(match["longitude"]),
            "timezone": match.get("timezone", "UTC"),
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
) -> Dict[str, float]:
    """Fetch current weather at an airport using Open-Meteo forecast API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m,precipitation,visibility",
        "wind_speed_unit": "kmh",
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
