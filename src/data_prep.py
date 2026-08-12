import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PrepConfig:
    flights_csv_path: str
    max_rows: Optional[int]
    delay_threshold_minutes: float


def _parse_time_hhmm(value: str) -> Optional[int]:
    """
    Convert CRS_DEP_TIME-like HHMM strings/ints to hour of day [0, 23].
    Returns None for invalid or missing values.
    """
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


def load_and_prepare_flights(config: PrepConfig) -> pd.DataFrame:
    """Load the raw on-time performance CSV and prepare core features and label."""
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

    df = pd.read_csv(
        config.flights_csv_path,
        nrows=config.max_rows,
        usecols=usecols,
    )

    # Filter to completed flights only (no cancellations/diversions)
    df = df[(df["CANCELLED"] == 0.0) & (df["DIVERTED"] == 0.0)].copy()

    # Define binary delay label based on ARR_DELAY
    df["delayed"] = (df["ARR_DELAY"] >= config.delay_threshold_minutes).astype(int)

    # Basic datetime from FL_DATE (format in sample: '1/1/2025 12:00:00 AM')
    df["flight_date"] = pd.to_datetime(df["FL_DATE"])

    # Departure hour from scheduled departure time
    df["dep_hour"] = df["CRS_DEP_TIME"].apply(_parse_time_hhmm)

    # Simple route identifier
    df["route"] = df["ORIGIN_AIRPORT_ID"].astype(str) + "_" + df["DEST_AIRPORT_ID"].astype(str)

    # Select model features + label
    feature_cols = [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "OP_UNIQUE_CARRIER",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "ORIGIN_STATE_ABR",
        "DEST_AIRPORT_ID",
        "DEST_STATE_ABR",
        "DEST",
        "dep_hour",
        "route",
    ]

    df_model = df[feature_cols + ["flight_date", "delayed", "ORIGIN_CITY_NAME"]].copy()
    return df_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare flight on-time data for modeling.")
    parser.add_argument("--csv", required=True, help="Path to T_ONTIME_REPORTING.csv")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional max number of rows to load (for quicker experiments).",
    )
    parser.add_argument(
        "--delay_threshold",
        type=float,
        default=15.0,
        help="Arrival delay in minutes to define a delayed flight.",
    )
    parser.add_argument(
        "--output_parquet",
        required=True,
        help="Output parquet file path for the prepared dataset.",
    )
    args = parser.parse_args()

    config = PrepConfig(
        flights_csv_path=args.csv,
        max_rows=args.max_rows,
        delay_threshold_minutes=args.delay_threshold,
    )
    df = load_and_prepare_flights(config)
    df.to_parquet(args.output_parquet, index=False)


if __name__ == "__main__":
    main()

