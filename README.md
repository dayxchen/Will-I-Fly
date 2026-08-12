## Flight delay prediction project

This project trains a machine learning model to predict whether a given **domestic flight** will be **delayed on arrival**, using:

- Historical **on-time performance** data (`T_ONTIME_REPORTING.csv`)
- **Weather data** from your preferred weather API

The target label is a binary flag indicating whether the **arrival delay** is greater than or equal to a configured threshold (default: **15 minutes**).

### 1. Setup

From the project root (`Will I Fly?`):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
```

### 2. Configure paths and weather API

Edit `src/config.yaml`:

- **`data.flights_csv_path`**: path to your `T_ONTIME_REPORTING.csv` file (relative or absolute).
- **`data.max_rows`**: optional cap on the number of rows to use during development (helps limit API calls).
- **`weather.base_url`**: base URL for your weather API endpoint.
- **`weather.api_key_env`**: name of the environment variable that will hold your API key.
- **`weather.query_params`**: query parameter names for airport, date, hour, and API key.
- **`weather.feature_fields`**: list of field names to extract from the API JSON response.

Export your weather API key in your shell, matching `weather.api_key_env`:

```bash
export WEATHER_API_KEY="your_real_api_key_here"
```

### 3. Data preparation (optional standalone step)

You can run flight-only preparation and inspect the intermediate parquet:

```bash
python src/data_prep.py \
  --csv "T_ONTIME_REPORTING.csv" \
  --max_rows 50000 \
  --delay_threshold 15 \
  --output_parquet "data/flights_prepared.parquet"
```

This will:

- Filter out cancelled and diverted flights.
- Create a `delayed` label based on `ARR_DELAY >= delay_threshold`.
- Engineer features like `dep_hour` and `route`.

### 4. Weather fetching and caching

The main training script automatically:

1. Extracts unique `(date, origin_airport_id, dep_hour)` keys from the flight data.
2. Fetches weather for each key from your API.
3. Stores the results in `data/weather_cache.csv` so subsequent runs reuse existing data without re-calling the API.

If you want to build the weather cache manually from the prepared parquet:

```bash
python src/weather_fetch.py \
  --flights_parquet "data/flights_prepared.parquet" \
  --base_url "https://your-weather-api.example.com/forecast" \
  --feature_fields "temperature,wind_speed,precipitation,visibility" \
  --cache_csv "data/weather_cache.csv"
```

You will likely need to adjust `--base_url`, `--feature_fields`, and the extraction logic inside `weather_fetch.py` to match your API’s schema.

### 5. Train the model end-to-end

With `src/config.yaml` updated and your API key set, run:

```bash
python src/train_model.py --config src/config.yaml --model_output models/flight_delay_model.pkl
```

The script will:

- Load and prepare the flight data.
- Merge in weather features using `data/weather_cache.csv` (creating or updating it as needed).
- Split the data into train/test sets.
- Train a `RandomForestClassifier`.
- Print metrics (classification report and ROC-AUC).
- Save the full preprocessing pipeline + model to `models/flight_delay_model.pkl`.

### 6. Using the trained model

`models/flight_delay_model.pkl` is a **scikit-learn pipeline**. To use it in your own code:

```python
import joblib
import pandas as pd

model = joblib.load("models/flight_delay_model.pkl")

# Build a single-row DataFrame with the same feature columns used in training.
# You must also attach corresponding weather features for that flight.
sample = pd.DataFrame([{
    "YEAR": 2025,
    "MONTH": 1,
    "DAY_OF_MONTH": 1,
    "DAY_OF_WEEK": 3,
    "OP_UNIQUE_CARRIER": "AA",
    "OP_CARRIER_FL_NUM": 1002,
    "ORIGIN_AIRPORT_ID": 13485,
    "ORIGIN_STATE_ABR": "WI",
    "DEST_AIRPORT_ID": 11057,
    "DEST_STATE_ABR": "NC",
    "DEST": "CLT",
    "dep_hour": 6,
    "route": "13485_11057",
    "temperature": 20.0,
    "wind_speed": 5.0,
    "precipitation": 0.0,
    "visibility": 10.0,
}])

pred_proba = model.predict_proba(sample)[:, 1][0]
print(f"Predicted probability of delay: {pred_proba:.3f}")
```

You can wrap this in an API or UI to get real-time predictions for upcoming flights, as long as you can supply the same types of features (flight metadata + weather at departure time).

---

### 7. Prediction API for your website

If you're building a **website** and want to own the UI/UX, use the small HTTP API so your frontend can get predictions without touching Python.

- **Next steps (what to run and when, API contract for your UI):** see **[NEXT_STEPS.md](NEXT_STEPS.md)**.
- Start the API server (after training once):
  ```bash
  uvicorn api:app --reload --app-dir src --host 0.0.0.0
  ```
  Then open `http://localhost:8000/docs` for the interactive API docs. Your site calls `POST /predict` with the same feature fields as above.

