# Will I Fly?

**Will I Fly?** tells you how likely your flight is to arrive late. Type in a flight number (for example `AA1002`), and the app looks up that flight, checks the current weather at the departure airport, and uses a machine learning model to estimate the chance it will land **15 or more minutes late**. It also explains which factors pushed the prediction up or down.

The app covers **US domestic flights**.

---

## How it works

```
 Flight number ──► Live flight data ──► Weather at origin ──► ML model ──► Delay probability
   "AA1002"         (Aviationstack)       (Open-Meteo)       (Random Forest)   + explanation (SHAP)
```

1. **You enter a flight number.** The app parses it into an airline code and number (`AA` + `1002`).
2. **It finds the flight.** The backend asks the [Aviationstack](https://aviationstack.com) API for the flight's current or next leg: airline, origin, destination, and scheduled departure time.
3. **It checks the weather.** Using the origin airport's location, it gets current temperature, wind speed, precipitation, and visibility from [Open-Meteo](https://open-meteo.com).
4. **The model makes a prediction.** The flight details and weather go into a trained Random Forest model, which returns the probability of a delay. Anything at 50% or above is labeled "likely delayed."
5. **It explains the result.** [SHAP](https://shap.readthedocs.io) breaks the prediction down into the factors that mattered most, such as the route, time of day, airline, or weather, and whether each one made a delay more or less likely.

### How the model was trained

- **Flight history:** US Department of Transportation on-time performance data (`T_ONTIME_REPORTING.csv`). Cancelled and diverted flights are removed, and each flight is labeled delayed if it arrived 15+ minutes late.
- **Historical weather:** For each airport, date, and departure hour, the matching weather is pulled from Open-Meteo's historical API and cached in `data/weather_cache.csv`, so it only has to be fetched once.
- **Features used:** airline, flight number, origin and destination airports and states, route, date (year, month, day, day of week), scheduled departure hour, and the four weather values.
- **Model:** a scikit-learn Random Forest (100 trees, max depth 20), tested on later dates than it was trained on so the scores reflect how it does on future flights. The trained pipeline is saved to `models/flight_delay_model.pkl`.

---

## Project layout

| Path | What it does |
| --- | --- |
| `src/api.py` | FastAPI backend that the website calls |
| `src/live_flights.py` | Looks up live flights through Aviationstack |
| `src/airport_geo.py` | Airport codes and locations, plus live weather |
| `src/flight_parser.py` | Turns text like `"AA 1002"` into airline + number |
| `src/flight_lookup.py` | Converts a flight record into model inputs |
| `src/shap_explain.py` | Builds the "why" explanation for a prediction |
| `src/data_prep.py` | Cleans the flight history and creates the delay label |
| `src/weather_fetch.py` | Fetches and caches historical weather for training |
| `src/train_model.py` | Trains and saves the model |
| `src/compress_model.py` | Shrinks the saved model for deployment |
| `src/config.yaml` | Settings for data paths, APIs, and training |
| `frontend/` | Helper code for calling the API from the React frontend |
| `models/` | Trained model files (stored with Git LFS) |
| `vercel.json` | Deploys the API to Vercel |

---

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/flights/lookup?flight=AA1002` | Main endpoint: finds a live flight and returns its delay prediction |
| `POST` | `/predict` | Predicts from flight and weather details you supply yourself |
| `POST` | `/predict/explain` | Same inputs as `/predict`, returns the SHAP explanation |
| `GET` | `/health` | Shows whether the model is loaded and the flight API key is set |

Interactive docs are at `/docs` when the server is running.

---

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Needed for live flight lookups
export AVIATIONSTACK_API_KEY="your_key"

# Start the API at http://localhost:8000
uvicorn api:app --reload --app-dir src --host 0.0.0.0
```

To retrain the model (Open-Meteo needs no API key):

```bash
python src/train_model.py --config src/config.yaml --model_output models/flight_delay_model.pkl
```

For connecting a React frontend, see [frontend/README.md](frontend/README.md) and [NEXT_STEPS.md](NEXT_STEPS.md).
