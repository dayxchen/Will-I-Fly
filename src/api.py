"""

Minimal prediction API for the flight delay model.

Run with: uvicorn api:app --reload --app-dir src

Your website UI can call POST /predict with the request body below.

"""

import os
import sys

# Ensure local modules in src/ resolve on Vercel (cwd may be project root).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from typing import Any, Dict, List, Optional



import joblib

import pandas as pd

import requests

import yaml

from fastapi import FastAPI, HTTPException, Query

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field



from flight_parser import parse_flight_number

from live_flights import LiveFlightService

from shap_explain import explain_prediction



# Load model once at startup (set MODEL_PATH env var or default)

MODEL_PATH = os.getenv("FLIGHT_DELAY_MODEL_PATH", "models/flight_delay_model.pkl")

_app_dir = Path(__file__).resolve().parent

_project_root = _app_dir.parent

_default_model = _project_root / "models" / "flight_delay_model.pkl"

_model_path = Path(MODEL_PATH) if os.path.isabs(MODEL_PATH) else _project_root / MODEL_PATH

_config_path = _app_dir / "config.yaml"



app = FastAPI(title="Flight delay prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://will-i-fly-frontend.vercel.app",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None

_live_flight_service: Optional[LiveFlightService] = None





def _load_app_config() -> dict:

    if _config_path.exists():

        with open(_config_path, "r") as f:

            return yaml.safe_load(f) or {}

    return {}





def get_live_flight_service() -> LiveFlightService:

    global _live_flight_service

    if _live_flight_service is None:

        cfg = _load_app_config().get("flights", {})

        api_key_env = cfg.get("api_key_env", "AVIATIONSTACK_API_KEY")

        _live_flight_service = LiveFlightService(

            api_key=os.getenv(api_key_env),

            api_url=cfg.get("api_url", "http://api.aviationstack.com/v1/flights"),

        )

    return _live_flight_service





def get_model():

    global _model

    if _model is None:

        path = _model_path if _model_path.exists() else _default_model

        if not path.exists():

            raise FileNotFoundError(

                f"No trained model at {path}. Train first: python src/train_model.py --config src/config.yaml"

            )

        _model = joblib.load(path)

    return _model





# Request body: same features as in train_model (categorical + numeric + weather).

# Your frontend can collect flight info + weather and send this.

class PredictRequest(BaseModel):

    # Categorical (strings)

    OP_UNIQUE_CARRIER: str = Field(..., description="Carrier code, e.g. AA")

    ORIGIN_STATE_ABR: str = Field(..., description="Origin state, e.g. NY")

    DEST_STATE_ABR: str = Field(..., description="Destination state, e.g. CA")

    DEST: str = Field(..., description="Destination airport code, e.g. LAX")

    route: str = Field(..., description="OriginAirportId_DestAirportId, e.g. 12478_12892")

    # Numeric

    YEAR: int = Field(..., ge=2000, le=2100)

    MONTH: int = Field(..., ge=1, le=12)

    DAY_OF_MONTH: int = Field(..., ge=1, le=31)

    DAY_OF_WEEK: int = Field(..., ge=1, le=7)

    OP_CARRIER_FL_NUM: int = Field(..., ge=1)

    ORIGIN_AIRPORT_ID: int = Field(...)

    DEST_AIRPORT_ID: int = Field(...)

    dep_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0-23)")

    # Weather (names must match config.yaml feature_fields)

    temperature: float = Field(..., description="Temperature at origin in Celsius")

    wind_speed: float = Field(..., ge=0, description="Wind speed in km/h")

    precipitation: float = Field(..., ge=0, description="Precipitation in mm")

    visibility: float = Field(..., ge=0, description="Visibility in meters")





class ShapFactor(BaseModel):

    feature: str

    impact: float

    direction: str = Field(..., description="increases_delay or decreases_delay")





class PredictionExplanation(BaseModel):

    base_probability_delayed: float

    summary: str

    factors: List[ShapFactor]





class PredictResponse(BaseModel):

    probability_delayed: float = Field(..., description="P(delay >= threshold), 0-1")

    delayed: bool = Field(..., description="True if probability_delayed >= 0.5")

    explanation: Optional[PredictionExplanation] = None





class PredictionResult(BaseModel):

    probability_delayed: float

    delayed: bool

    model_available: bool = True

    explanation: Optional[PredictionExplanation] = None





class FlightWithPrediction(BaseModel):

    flight: Dict[str, Any]

    prediction: Optional[PredictionResult] = None





class FlightLookupResponse(BaseModel):

    query: str

    carrier: str

    flight_number: int

    count: int

    flights: List[FlightWithPrediction]





def _prepare_predict_df(body: Dict[str, Any]) -> pd.DataFrame:

    df = pd.DataFrame([body])

    for col in ["OP_UNIQUE_CARRIER", "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "DEST", "route"]:

        df[col] = df[col].astype(str)

    return df





def _build_explanation(model, df: pd.DataFrame, delayed: bool) -> Optional[PredictionExplanation]:

    try:

        payload = explain_prediction(model, df, delayed=delayed)

        return PredictionExplanation(**payload)

    except Exception:

        return None





def _run_prediction(flight: Dict[str, Any]) -> Optional[PredictionResult]:

    try:

        model = get_model()

    except FileNotFoundError:

        return None



    try:

        body = flight.get("predict_request")

        if not body:

            return None

    except ValueError:

        return None



    df = _prepare_predict_df(body)

    proba = float(model.predict_proba(df)[0, 1])

    delayed = proba >= 0.5

    return PredictionResult(

        probability_delayed=round(proba, 4),

        delayed=delayed,

        model_available=True,

    )





@app.get("/")

def root():

    return {

        "message": "Flight delay prediction API",

        "docs": "/docs",

        "predict": "POST /predict",

        "lookup": "GET /flights/lookup?flight=AA1002",

        "mode": "live",

    }





@app.get("/health")

def health():

    model_loaded = False

    model_detail = None

    try:

        get_model()

        model_loaded = True

    except Exception as e:

        model_detail = str(e)



    flight_api_configured = bool(os.getenv("AVIATIONSTACK_API_KEY"))

    return {

        "status": "ok",

        "mode": "live",

        "model_loaded": model_loaded,

        "flight_api_configured": flight_api_configured,

        "model_detail": model_detail,

    }





@app.get("/flights/lookup", response_model=FlightLookupResponse)

def lookup_flight(

    flight: str = Query(..., description="Flight number, e.g. AA1002 or AA 1002"),

):

    try:

        carrier, flight_num = parse_flight_number(flight)

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))



    try:

        service = get_live_flight_service()

        record = service.lookup_current_leg(flight)

    except RuntimeError as e:

        raise HTTPException(status_code=503, detail=str(e))

    except LookupError as e:

        raise HTTPException(status_code=404, detail=str(e))

    except requests.HTTPError as e:

        raise HTTPException(status_code=502, detail=f"Flight data provider error: {e}")



    enriched = [

        FlightWithPrediction(

            flight=record,

            prediction=_run_prediction(record),

        )

    ]



    return FlightLookupResponse(

        query=flight.strip(),

        carrier=carrier,

        flight_number=flight_num,

        count=1,

        flights=enriched,

    )





@app.post("/predict", response_model=PredictResponse)

def predict(body: PredictRequest):

    try:

        model = get_model()

    except FileNotFoundError as e:

        raise HTTPException(status_code=503, detail=str(e))

    row = body.model_dump()

    df = _prepare_predict_df(row)

    proba = float(model.predict_proba(df)[0, 1])

    delayed = proba >= 0.5

    return PredictResponse(

        probability_delayed=round(proba, 4),

        delayed=delayed,

    )





@app.post("/predict/explain", response_model=PredictionExplanation)

def explain(body: PredictRequest):

    try:

        model = get_model()

    except FileNotFoundError as e:

        raise HTTPException(status_code=503, detail=str(e))

    df = _prepare_predict_df(body.model_dump())

    proba = float(model.predict_proba(df)[0, 1])

    delayed = proba >= 0.5

    explanation = _build_explanation(model, df, delayed)

    if explanation is None:

        raise HTTPException(status_code=500, detail="Could not generate SHAP explanation.")

    return explanation


