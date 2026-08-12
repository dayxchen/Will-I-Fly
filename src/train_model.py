import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import PrepConfig, load_and_prepare_flights
from weather_fetch import WeatherConfig, build_weather_keys, build_weather_table


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end preparation: flights + weather -> X, y."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    weather_cfg = cfg["weather"]

    prep_config = PrepConfig(
        flights_csv_path=data_cfg["flights_csv_path"],
        max_rows=data_cfg.get("max_rows"),
        delay_threshold_minutes=train_cfg["delay_threshold_minutes"],
    )
    flights_df = load_and_prepare_flights(prep_config)

    # Build / load weather table
    weather_cache_path = os.path.join("data", "weather_cache.csv")
    weather_config = WeatherConfig(
        archive_url=weather_cfg["archive_url"],
        geocoding_url=weather_cfg["geocoding_url"],
        feature_fields=weather_cfg["feature_fields"],
        hourly_variables=weather_cfg.get(
            "hourly_variables",
            [
                "temperature_2m",
                "wind_speed_10m",
                "precipitation",
                "visibility",
            ],
        ),
        coords_cache_path=os.path.join("data", "airport_coords_cache.csv"),
        request_delay_seconds=weather_cfg.get("request_delay_seconds", 0.1),
    )

    keys = build_weather_keys(flights_df)
    weather_df = build_weather_table(keys, weather_config, cache_path=weather_cache_path)

    # Join weather onto flights
    weather_df_subset = weather_df.copy()
    join_cols = ["date_str", "ORIGIN_AIRPORT_ID", "dep_hour"]
    flights_df = flights_df.copy()
    flights_df["date_str"] = flights_df["flight_date"].dt.strftime("%Y-%m-%d")
    merged = flights_df.merge(
        weather_df_subset,
        on=join_cols,
        how="left",
        validate="m:1",
    )

    # Drop rows with missing critical weather values
    merged = merged.dropna(subset=weather_cfg["feature_fields"])

    y = merged["delayed"].astype(int)

    # Feature columns (categorical + numeric)
    categorical_cols = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN_STATE_ABR",
        "DEST_STATE_ABR",
        "DEST",
        "route",
    ]
    numeric_cols = [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID",
        "dep_hour",
    ] + weather_cfg["feature_fields"]

    X = merged[categorical_cols + numeric_cols].copy()
    X[categorical_cols] = X[categorical_cols].astype(str)

    return X, y, categorical_cols, numeric_cols


def build_pipeline(categorical_cols, numeric_cols) -> Pipeline:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )
    return model


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    model_output_path: str,
) -> None:
    train_cfg = cfg["training"]

    if train_cfg.get("time_based_split", True):
        # For simplicity, fall back to random split; you can replace this
        # with a proper time-based split using flight_date if desired.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_cfg["test_size"], random_state=train_cfg["random_state"], stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_cfg["test_size"], random_state=train_cfg["random_state"], stratify=y
        )

    model = build_pipeline(
        categorical_cols=[c for c in X.columns if X[c].dtype == "object"],
        numeric_cols=[c for c in X.columns if X[c].dtype != "object"],
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        auc = None

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    if auc is not None:
        print(f"ROC-AUC: {auc:.3f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Saved trained model to {model_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flight delay prediction model.")
    parser.add_argument(
        "--config",
        default="src/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model_output",
        default="models/flight_delay_model.pkl",
        help="Where to save the trained model.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    X, y, cat_cols, num_cols = prepare_dataset(cfg)
    train_and_evaluate(X, y, cfg, args.model_output)


if __name__ == "__main__":
    main()

