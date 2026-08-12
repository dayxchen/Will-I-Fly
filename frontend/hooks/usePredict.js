import { useState, useCallback } from 'react';
import { predictDelay } from '../api/predict';

/**
 * React hook to call the prediction API.
 * Copy into your app (e.g. src/hooks/usePredict.js) and fix the import path to your api/predict.
 *
 * Usage:
 *   const { predict, loading, result, error } = usePredict();
 *   const run = () => predict({ OP_UNIQUE_CARRIER: 'AA', ... });
 *   // result => { probability_delayed: 0.32, delayed: false }
 */
export function usePredict() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const predict = useCallback(async (body) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictDelay(body);
      setResult(data);
      return data;
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  return { predict, loading, result, error };
}
