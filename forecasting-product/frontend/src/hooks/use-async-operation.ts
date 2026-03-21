"use client";

import { useState, useCallback } from "react";

/**
 * Generic hook for async operations with loading/error/result state.
 * Eliminates repeated try/catch/finally patterns across pages.
 */
export function useAsyncOperation<T>() {
  const [result, setResult] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (fn: () => Promise<T>) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fn();
      setResult(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Operation failed";
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, run, reset, setError };
}
