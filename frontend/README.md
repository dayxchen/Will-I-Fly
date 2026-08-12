# Frontend – add this to your React app

Your React app runs at **http://localhost:3000**. To get predictions from the Python API, add the following.

## 1. Environment variable

In your React project root, create or edit **`.env`** (and add `.env` to `.gitignore` if it isn’t already):

```bash
# Create React App
REACT_APP_API_URL=http://localhost:8000

# If you use Vite instead:
# VITE_API_URL=http://localhost:8000
```

Restart the dev server after changing `.env`.

## 2. Copy these into your React app

- **`api/predict.js`** → copy to your app’s `src/api/predict.js` (or `src/services/predict.js`).
- **`hooks/usePredict.js`** → copy to your app’s `src/hooks/usePredict.js`, then fix the import inside it to point at your `api/predict` file (e.g. `from '../api/predict'` or `from '../services/predict'`).

You can build your own UI and call the API in either way:

- **Direct:** `import { predictDelay } from './api/predict';` then `const data = await predictDelay(body);`
- **Hook:** `const { predict, loading, result, error } = usePredict();` then call `predict(body)` from a button or form submit.

## 3. Request body shape

`body` for `predictDelay(body)` or `predict(body)` must match what the backend expects. From the project root see **NEXT_STEPS.md** for the full list. Minimal example:

```js
const body = {
  OP_UNIQUE_CARRIER: 'AA',
  ORIGIN_STATE_ABR: 'NY',
  DEST_STATE_ABR: 'CA',
  DEST: 'LAX',
  route: '12478_12892',
  YEAR: 2025,
  MONTH: 1,
  DAY_OF_MONTH: 15,
  DAY_OF_WEEK: 3,
  OP_CARRIER_FL_NUM: 100,
  ORIGIN_AIRPORT_ID: 12478,
  DEST_AIRPORT_ID: 12892,
  dep_hour: 14,
  temperature: 20,
  wind_speed: 5,
  precipitation: 0,
  visibility: 10,
};
```

Response: `{ probability_delayed: number, delayed: boolean }`.

## 4. Run the backend when you need predictions

From the **“Will I Fly?”** project root:

```bash
source .venv/bin/activate
uvicorn api:app --reload --app-dir src --host 0.0.0.0
```

Then the API is at **http://localhost:3000** (React) and **http://localhost:8000** (API). Your UI calls `http://localhost:8000/predict` (or uses the env var so it works in production too).
