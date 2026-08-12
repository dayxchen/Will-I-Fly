"""Fetch live flight data by flight number."""
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import requests

from airport_geo import AirportGeoCache, fetch_current_weather, iata_to_airport_id
from flight_parser import parse_flight_number
from flight_lookup import flight_to_predict_request

LIVE_STATUS_PRIORITY = {
    "active": 0,
    "scheduled": 1,
    "landed": 50,
    "cancelled": 100,
    "diverted": 100,
    "incident": 100,
}

CURRENT_LEG_STATUSES = {"active", "scheduled"}


class LiveFlightService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "http://api.aviationstack.com/v1/flights",
        airport_geo: Optional[AirportGeoCache] = None,
    ):
        self.api_key = api_key or os.getenv("AVIATIONSTACK_API_KEY")
        self.api_url = api_url
        self.airport_geo = airport_geo or AirportGeoCache()
        self.session = requests.Session()

    def lookup_current_leg(self, query: str) -> Dict[str, Any]:
        carrier, flight_num = parse_flight_number(query)
        flight_iata = f"{carrier}{flight_num}"

        if not self.api_key:
            raise RuntimeError(
                "AVIATIONSTACK_API_KEY is not set. Sign up at https://aviationstack.com "
                "and export your access key before searching live flights."
            )

        raw_flights = self._fetch_flights(flight_iata)
        if not raw_flights:
            raise LookupError(
                f"No live data found for {flight_iata}. The flight may not be operating today."
            )

        selected = self._select_current_leg(raw_flights)
        if selected is None:
            raise LookupError(
                f"No active or scheduled leg found for {flight_iata} right now."
            )

        normalized = self._normalize_flight(selected, carrier, flight_num)
        self.airport_geo.save()
        return normalized

    def _fetch_flights(self, flight_iata: str) -> List[Dict[str, Any]]:
        # Free Aviationstack plans do not allow the flight_date filter (403).
        # Fetch by flight number and filter to today's leg in _select_current_leg.
        params = {
            "access_key": self.api_key,
            "flight_iata": flight_iata,
            "limit": 100,
        }
        payload = self._request_flights(params)
        return payload.get("data") or []

    def _request_flights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.get(self.api_url, params=params, timeout=20)
        try:
            payload = resp.json()
        except ValueError:
            payload = {}

        if payload.get("error"):
            error = payload["error"]
            code = error.get("code", "")
            message = error.get("message") or "Aviationstack API error"
            if code == "function_access_restricted":
                message = (
                    f"{message} Try upgrading your Aviationstack plan, or remove "
                    "unsupported query parameters."
                )
            elif code == "https_access_restricted":
                message = (
                    f"{message} Free plans must use http://api.aviationstack.com "
                    "(not https)."
                )
            raise RuntimeError(message)

        if not resp.ok:
            raise RuntimeError(
                f"Flight data provider error ({resp.status_code}): "
                f"{payload.get('error', {}).get('message', resp.reason)}"
            )

        return payload

    def _select_current_leg(self, flights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        today = date.today().isoformat()
        candidates = [f for f in flights if f.get("flight_date") == today] or flights

        candidates = [
            f
            for f in candidates
            if f.get("flight_status") not in {"cancelled", "diverted", "incident"}
        ]
        if not candidates:
            return None

        live_candidates = [
            f for f in candidates if f.get("flight_status") in CURRENT_LEG_STATUSES
        ]
        if live_candidates:
            candidates = live_candidates

        candidates.sort(
            key=lambda f: (
                LIVE_STATUS_PRIORITY.get(f.get("flight_status"), 25),
                f.get("departure", {}).get("scheduled") or "",
            )
        )
        return candidates[0]

    def _normalize_flight(
        self,
        raw: Dict[str, Any],
        carrier: str,
        flight_num: int,
    ) -> Dict[str, Any]:
        departure = raw.get("departure") or {}
        arrival = raw.get("arrival") or {}
        airline = raw.get("airline") or {}
        flight_meta = raw.get("flight") or {}
        live = raw.get("live") or None
        aircraft = raw.get("aircraft") or None

        origin_iata = departure.get("iata") or ""
        dest_iata = arrival.get("iata") or ""
        if not origin_iata or not dest_iata:
            raise LookupError("Live flight is missing origin or destination airport codes.")

        origin_geo = self.airport_geo.resolve(
            origin_iata,
            departure.get("airport") or origin_iata,
            self.session,
        )
        dest_geo = self.airport_geo.resolve(
            dest_iata,
            arrival.get("airport") or dest_iata,
            self.session,
        )
        if not origin_geo or not dest_geo:
            raise LookupError("Could not resolve airport location for this flight.")

        scheduled_dep = departure.get("scheduled")
        dep_dt = _parse_iso_datetime(scheduled_dep, departure.get("timezone"))
        dep_hour = dep_dt.hour if dep_dt else None

        origin_id = int(origin_geo["airport_id"])
        dest_id = int(dest_geo["airport_id"])
        dep_delay = _safe_int(departure.get("delay"))
        arr_delay = _safe_int(arrival.get("delay"))

        weather = fetch_current_weather(
            float(origin_geo["latitude"]),
            float(origin_geo["longitude"]),
            self.session,
        )

        flight = {
            "source": "live",
            "flight_iata": f"{carrier}{flight_num}",
            "OP_UNIQUE_CARRIER": carrier,
            "OP_CARRIER_FL_NUM": flight_num,
            "flight_status": raw.get("flight_status"),
            "flight_date": raw.get("flight_date"),
            "airline_name": airline.get("name"),
            "aircraft_registration": (aircraft or {}).get("registration"),
            "aircraft_iata": (aircraft or {}).get("iata"),
            "origin_iata": origin_iata,
            "origin_name": departure.get("airport"),
            "origin_terminal": departure.get("terminal"),
            "origin_gate": departure.get("gate"),
            "origin_timezone": departure.get("timezone"),
            "destination_iata": dest_iata,
            "destination_name": arrival.get("airport"),
            "destination_terminal": arrival.get("terminal"),
            "destination_gate": arrival.get("gate"),
            "destination_timezone": arrival.get("timezone"),
            "departure_scheduled": scheduled_dep,
            "departure_estimated": departure.get("estimated"),
            "departure_actual": departure.get("actual"),
            "departure_delay_minutes": dep_delay,
            "arrival_scheduled": arrival.get("scheduled"),
            "arrival_estimated": arrival.get("estimated"),
            "arrival_actual": arrival.get("actual"),
            "arrival_delay_minutes": arr_delay,
            "currently_delayed": _is_currently_delayed(raw.get("flight_status"), dep_delay, arr_delay),
            "live": live,
            "weather": weather,
            # Model feature fields
            "YEAR": dep_dt.year if dep_dt else datetime.utcnow().year,
            "MONTH": dep_dt.month if dep_dt else datetime.utcnow().month,
            "DAY_OF_MONTH": dep_dt.day if dep_dt else datetime.utcnow().day,
            "DAY_OF_WEEK": dep_dt.isoweekday() if dep_dt else datetime.utcnow().isoweekday(),
            "ORIGIN_AIRPORT_ID": origin_id,
            "ORIGIN_CITY_NAME": departure.get("airport"),
            "ORIGIN_STATE_ABR": origin_geo["state_abr"],
            "DEST_AIRPORT_ID": dest_id,
            "DEST_CITY_NAME": arrival.get("airport"),
            "DEST_STATE_ABR": dest_geo["state_abr"],
            "DEST": dest_iata,
            "route": f"{origin_id}_{dest_id}",
            "dep_hour": dep_hour,
        }
        flight["predict_request"] = flight_to_predict_request(flight, weather=weather)
        return flight


def _parse_iso_datetime(value: Optional[str], timezone: Optional[str] = None) -> Optional[datetime]:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_currently_delayed(status: Optional[str], dep_delay: Optional[int], arr_delay: Optional[int]) -> bool:
    if status == "active" and arr_delay is not None:
        return arr_delay >= 15
    if dep_delay is not None:
        return dep_delay >= 15
    if arr_delay is not None:
        return arr_delay >= 15
    return False


def format_display_time(value: Optional[str], timezone: Optional[str] = None) -> Optional[str]:
    dt = _parse_iso_datetime(value, timezone)
    if not dt:
        return value
    if timezone:
        return dt.strftime("%b %d, %Y %H:%M %Z")
    return dt.strftime("%b %d, %Y %H:%M UTC")
