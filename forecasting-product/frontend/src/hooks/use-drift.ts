"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useDrift(
  lob: string,
  params?: { run_type?: string; baseline_weeks?: number; recent_weeks?: number },
) {
  return useQuery({
    queryKey: ["drift", lob, params],
    queryFn: () => api.getDrift(lob, params),
    enabled: !!lob,
  });
}
