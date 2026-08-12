/**
 * Call the flight delay prediction API.
 * Copy this file into your React app (e.g. src/api/predict.js).
 *
 * Expects .env with REACT_APP_API_URL (CRA) or VITE_API_URL (Vite).
 */

const getBaseUrl = () => {
  if (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  return 'http://localhost:8000';
};

/**
 * Request body shape expected by POST /predict (see NEXT_STEPS.md).
 */
export async function predictDelay(body) {
  const base = getBaseUrl();
  const res = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}
