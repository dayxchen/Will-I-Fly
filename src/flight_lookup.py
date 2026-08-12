"""Look up flights from the on-time reporting CSV by carrier + flight number."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from flight_parser import parse_flight_number


def _parse_time_hhmm(value: str) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        s = str(int(value)).zfill(4)
        hour = int(s[:2])
        if 0 <= hour <= 23:
            return hour
    except Exception:
        return None
    return None

from flight_parser import parse_flight_number


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_csv_path() -> Path:
    return _project_root() / "T_ONTIME_REPORTING.csv"


class FlightLookup:
    """Cached CSV loader and flight search."""

    def __init__(self, csv_path: Optional[str] = None, delay_threshold: float = 15.0):
        self.csv_path = Path(csv_path) if csv_path else _default_csv_path()
        self.delay_threshold = delay_threshold
        self._df: Optional[pd.DataFrame] = None
        self._raw_df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._raw_df is not None:
            return self._raw_df

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Flight data not found at {self.csv_path}")

        usecols = [
            "YEAR",
            "MONTH",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "FL_DATE",
            "OP_UNIQUE_CARRIER",
            "OP_CARRIER_FL_NUM",
            "ORIGIN_AIRPORT_ID",
            "ORIGIN_CITY_NAME",
            "ORIGIN_STATE_ABR",
            "DEST_AIRPORT_ID",
            "DEST_CITY_NAME",
            "DEST_STATE_ABR",
            "DEST",
            "CRS_DEP_TIME",
            "DEP_TIME",
            "DEP_DELAY",
            "CRS_ARR_TIME",
            "ARR_TIME",
            "ARR_DELAY",
            "CANCELLED",
            "DIVERTED",
        ]

        df = pd.read_csv(self.csv_path, usecols=usecols)
        df = df[(df["CANCELLED"] == 0.0) & (df["DIVERTED"] == 0.0)].copy()
        df["flight_date"] = pd.to_datetime(df["FL_DATE"])
        df["dep_hour"] = df["CRS_DEP_TIME"].apply(_parse_time_hhmm)
        df["route"] = (
            df["ORIGIN_AIRPORT_ID"].astype(str) + "_" + df["DEST_AIRPORT_ID"].astype(str)
        )
        df["actual_delayed"] = (df["ARR_DELAY"] >= self.delay_threshold).astype(bool)
        self._raw_df = df
        return df

    def lookup(
        self, carrier: str, flight_num: int, limit: int = 5
    ) -> List[Dict[str, Any]]:
        df = self._load()
        mask = (df["OP_UNIQUE_CARRIER"] == carrier.upper()) & (
            df["OP_CARRIER_FL_NUM"] == flight_num
        )
        matches = df.loc[mask].sort_values("flight_date", ascending=False).head(limit)
        if matches.empty:
            return []

        results: List[Dict[str, Any]] = []
        for _, row in matches.iterrows():
            results.append(_row_to_flight_dict(row))
        return results


def _row_to_flight_dict(row: pd.Series) -> Dict[str, Any]:
    dep_hour = row["dep_hour"]
    return {
        "YEAR": int(row["YEAR"]),
        "MONTH": int(row["MONTH"]),
        "DAY_OF_MONTH": int(row["DAY_OF_MONTH"]),
        "DAY_OF_WEEK": int(row["DAY_OF_WEEK"]),
        "FL_DATE": row["flight_date"].strftime("%Y-%m-%d"),
        "OP_UNIQUE_CARRIER": str(row["OP_UNIQUE_CARRIER"]),
        "OP_CARRIER_FL_NUM": int(row["OP_CARRIER_FL_NUM"]),
        "ORIGIN_AIRPORT_ID": int(row["ORIGIN_AIRPORT_ID"]),
        "ORIGIN_CITY_NAME": str(row["ORIGIN_CITY_NAME"]),
        "ORIGIN_STATE_ABR": str(row["ORIGIN_STATE_ABR"]),
        "DEST_AIRPORT_ID": int(row["DEST_AIRPORT_ID"]),
        "DEST_CITY_NAME": str(row["DEST_CITY_NAME"]),
        "DEST_STATE_ABR": str(row["DEST_STATE_ABR"]),
        "DEST": str(row["DEST"]),
        "route": str(row["route"]),
        "CRS_DEP_TIME": _format_hhmm(row["CRS_DEP_TIME"]),
        "DEP_TIME": _format_hhmm(row["DEP_TIME"]),
        "DEP_DELAY": _safe_float(row["DEP_DELAY"]),
        "CRS_ARR_TIME": _format_hhmm(row["CRS_ARR_TIME"]),
        "ARR_TIME": _format_hhmm(row["ARR_TIME"]),
        "ARR_DELAY": _safe_float(row["ARR_DELAY"]),
        "dep_hour": int(dep_hour) if pd.notna(dep_hour) else None,
        "actual_delayed": bool(row["actual_delayed"]),
    }


def _format_hhmm(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    try:
        s = str(int(value)).zfill(4)
        return f"{s[:2]}:{s[2:]}"
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    return round(float(value), 2)


def flight_to_predict_request(
    flight: Dict[str, Any],
    weather: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a POST /predict body from a flight record."""
    defaults = {
        "temperature": 20.0,
        "wind_speed": 5.0,
        "precipitation": 0.0,
        "visibility": 10000.0,
    }
    if weather:
        defaults.update(weather)

    dep_hour = flight.get("dep_hour")
    if dep_hour is None:
        raise ValueError("Flight is missing departure hour.")

    return {
        "OP_UNIQUE_CARRIER": flight["OP_UNIQUE_CARRIER"],
        "ORIGIN_STATE_ABR": flight["ORIGIN_STATE_ABR"],
        "DEST_STATE_ABR": flight["DEST_STATE_ABR"],
        "DEST": flight["DEST"],
        "route": flight["route"],
        "YEAR": flight["YEAR"],
        "MONTH": flight["MONTH"],
        "DAY_OF_MONTH": flight["DAY_OF_MONTH"],
        "DAY_OF_WEEK": flight["DAY_OF_WEEK"],
        "OP_CARRIER_FL_NUM": flight["OP_CARRIER_FL_NUM"],
        "ORIGIN_AIRPORT_ID": flight["ORIGIN_AIRPORT_ID"],
        "DEST_AIRPORT_ID": flight["DEST_AIRPORT_ID"],
        "dep_hour": dep_hour,
        **defaults,
    }
