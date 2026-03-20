"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useLeaderboard(lob: string, runType: string = "backtest") {
  return useQuery({
    queryKey: ["leaderboard", lob, runType],
    queryFn: () => api.getLeaderboard(lob, runType),
    enabled: !!lob,
  });
}
