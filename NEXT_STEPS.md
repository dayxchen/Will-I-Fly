# Next steps (website + when to run things)

You’ve set the weather API in `config.yaml`. Here’s what to do next and when you need to run anything.

---

## Do you need to run anything right now?

**No.** If you’re only designing and building the UI/UX, you can:

- Build your website (forms, layout, copy) and call the **prediction API** when it’s ready.
- Run the API and training **later**, when you want real predictions.

When you’re ready for real predictions, you’ll (1) train the model once, then (2) run the API server. Your site then talks to that API.

---

## Order of operations when you’re ready

1. **Train the model (one-time)**  
   From the project root:
   ```bash
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   export WEATHER_API_KEY="your_key"   # if your API needs it (weather.gov often doesn’t)
   python src/train_model.py --config src/config.yaml --model_output models/flight_delay_model.pkl
   ```
   This uses your CSV + weather API, builds the model, and saves it to `models/flight_delay_model.pkl`.

2. **Run the prediction API**  
   So your website can get predictions:
   ```bash
   cd "/Users/davin/Documents/Will I Fly?"
   source .venv/bin/activate
   uvicorn api:app --reload --app-dir src --host 0.0.0.0
   ```
   Then:
   - API base URL: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - Health: `GET http://localhost:8000/health`

3. **Build your website**  
   Your UI only needs to:
   - Collect flight + weather inputs (or get weather from your API on the server).
   - Send **POST /predict** with the JSON body below.
   - Use the response to show “Will I fly?” (e.g. probability and delayed yes/no).

---

## API contract for your UI

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`

**Request body** (one flight + weather at origin):

```json
{
  "OP_UNIQUE_CARRIER": "AA",
  "ORIGIN_STATE_ABR": "NY",
  "DEST_STATE_ABR": "CA",
  "DEST": "LAX",
  "route": "12478_12892",
  "YEAR": 2025,
  "MONTH": 1,
  "DAY_OF_MONTH": 15,
  "DAY_OF_WEEK": 3,
  "OP_CARRIER_FL_NUM": 100,
  "ORIGIN_AIRPORT_ID": 12478,
  "DEST_AIRPORT_ID": 12892,
  "dep_hour": 14,
  "temperature": 20.0,
  "wind_speed": 5.0,
  "precipitation": 0.0,
  "visibility": 10.0
}
```

**Response:**

```json
{
  "probability_delayed": 0.32,
  "delayed": false
}
```

If the weather API in your config uses different field names than `temperature`, `wind_speed`, `precipitation`, `visibility`, either:

- Change `config.yaml` `weather.feature_fields` to match what you use when training, and keep the same names in the API and your UI, or  
- Add a mapping in your backend so the API still accepts the same request shape above.

---

## Summary

| Goal | Run now? |
|------|----------|
| Design/build the website UI/UX only | No |
| Get real predictions in the UI | Yes: train once, then run `uvicorn api:app --reload --app-dir src` and point your site at `POST /predict` |

You own the UI/UX; the repo gives you the model pipeline and this API so your frontend has a single endpoint to call.
