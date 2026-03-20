"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useAudit(params?: { action?: string; resource_type?: string; limit?: number }) {
  return useQuery({
    queryKey: ["audit", params],
    queryFn: () => api.getAuditLog(params),
  });
}
