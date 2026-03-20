"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useForecast(lob: string, params?: { series_id?: string; horizon?: number }) {
  return useQuery({
    queryKey: ["forecast", lob, params],
    queryFn: () => api.getForecasts(lob, params),
    enabled: !!lob,
  });
}

export function useForecastSeries(lob: string, seriesId: string) {
  return useQuery({
    queryKey: ["forecast", lob, seriesId],
    queryFn: () => api.getForecastSeries(lob, seriesId),
    enabled: !!lob && !!seriesId,
  });
}
