"""SHAP-based explanations for flight delay predictions."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from scipy.sparse import issparse

_explainer_cache: Dict[int, shap.TreeExplainer] = {}

DEFAULT_SHAP_MAX_TREES = int(os.getenv("SHAP_MAX_TREES", "25"))

FEATURE_LABELS = {
    "temperature": "Departure temperature",
    "wind_speed": "Wind speed at origin",
    "precipitation": "Precipitation at origin",
    "visibility": "Visibility at origin",
    "dep_hour": "Scheduled departure hour",
    "DAY_OF_WEEK": "Day of week",
    "DAY_OF_MONTH": "Day of month",
    "MONTH": "Month of year",
    "YEAR": "Year",
    "OP_CARRIER_FL_NUM": "Flight number",
    "ORIGIN_AIRPORT_ID": "Origin airport ID",
    "DEST_AIRPORT_ID": "Destination airport ID",
}

CATEGORICAL_PREFIXES = [
    ("OP_UNIQUE_CARRIER_", "Carrier"),
    ("ORIGIN_STATE_ABR_", "Origin state"),
    ("DEST_STATE_ABR_", "Destination state"),
    ("DEST_", "Destination airport"),
    ("route_", "Route"),
]


def _get_explainer(model, max_trees: Optional[int] = None) -> shap.TreeExplainer:
    if max_trees is None:
        max_trees = DEFAULT_SHAP_MAX_TREES

    cache_key = (id(model), max_trees)
    if cache_key not in _explainer_cache:
        clf = model.named_steps["model"]
        if max_trees and max_trees < len(clf.estimators_):
            clf = copy.copy(clf)
            clf.estimators_ = clf.estimators_[:max_trees]
            clf.n_estimators = len(clf.estimators_)
        _explainer_cache[cache_key] = shap.TreeExplainer(clf)
    return _explainer_cache[cache_key]


def _transform_features(model, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    preprocessor = model.named_steps["preprocess"]
    transformed = preprocessor.transform(df)
    if issparse(transformed):
        transformed = transformed.toarray()
    matrix = np.asarray(transformed, dtype=np.float64)
    feature_names = preprocessor.get_feature_names_out()
    return matrix, feature_names


def _format_feature_name(raw_name: str) -> str:
    if raw_name.startswith("cat__"):
        value = raw_name[5:]
        for prefix, label in CATEGORICAL_PREFIXES:
            if value.startswith(prefix):
                return f"{label}: {value[len(prefix):]}"
        return value.replace("_", " ")

    if raw_name.startswith("num__"):
        key = raw_name[5:]
        return FEATURE_LABELS.get(key, key.replace("_", " ").title())

    return raw_name.replace("_", " ")


def _extract_shap_row(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1][0], dtype=np.float64)

    arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        return arr[0, :, 1]
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def _base_delay_probability(explainer: shap.TreeExplainer) -> float:
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        values = np.asarray(expected, dtype=np.float64).ravel()
        return float(values[1] if values.size > 1 else values[0])
    return float(expected)


def _is_active_feature(feature_name: str, matrix_row: np.ndarray, idx: int) -> bool:
    """One-hot categoricals are only active for the flight's actual category."""
    if str(feature_name).startswith("cat__"):
        return matrix_row[idx] != 0.0
    return True


def _aggregate_factors(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate display labels (e.g. two carrier dummies) into one row."""
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for factor in factors:
        label = factor["feature"]
        if label not in merged:
            merged[label] = dict(factor)
            order.append(label)
            continue

        merged[label]["impact"] = round(merged[label]["impact"] + factor["impact"], 5)
        impact = merged[label]["impact"]
        merged[label]["direction"] = "increases_delay" if impact > 0 else "decreases_delay"

    return [merged[label] for label in order]


def _build_summary(factors: List[Dict[str, Any]], delayed: bool) -> str:
    if not factors:
        return "Not enough signal to explain this prediction."

    increases = [f["feature"] for f in factors if f["impact"] > 0][:2]
    decreases = [f["feature"] for f in factors if f["impact"] < 0][:2]

    if delayed:
        if increases:
            lead = f"Delay risk is driven mainly by {', '.join(increases)}."
        else:
            lead = "Delay risk is elevated despite several factors pointing toward on-time arrival."
        if decreases:
            return f"{lead} Working against a delay: {', '.join(decreases)}."
        return lead

    if decreases:
        lead = f"On-time outlook is supported by {', '.join(decreases)}."
    else:
        lead = "On-time outlook holds even with some factors adding delay risk."
    if increases:
        return f"{lead} Factors adding delay risk: {', '.join(increases)}."
    return lead


def explain_prediction(
    model,
    df: pd.DataFrame,
    delayed: bool,
    top_n: int = 6,
) -> Dict[str, Any]:
    """Return top SHAP contributors for a single-row prediction."""
    matrix, feature_names = _transform_features(model, df)
    matrix_row = matrix[0]
    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(matrix, check_additivity=False)
    row = _extract_shap_row(shap_values)

    ranked_idx = np.argsort(np.abs(row))[::-1]
    candidates: List[Dict[str, Any]] = []

    for idx in ranked_idx:
        feature_name = str(feature_names[idx])
        impact = float(row[idx])
        if impact == 0.0:
            continue
        if not _is_active_feature(feature_name, matrix_row, idx):
            continue

        candidates.append(
            {
                "feature": _format_feature_name(feature_name),
                "impact": round(impact, 5),
                "direction": "increases_delay" if impact > 0 else "decreases_delay",
            }
        )

    factors = _aggregate_factors(candidates)[:top_n]

    return {
        "base_probability_delayed": round(_base_delay_probability(explainer), 4),
        "summary": _build_summary(factors, delayed),
        "factors": factors,
    }
