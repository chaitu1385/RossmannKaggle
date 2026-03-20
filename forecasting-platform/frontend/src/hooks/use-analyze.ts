"use client";

import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useAnalyze() {
  return useMutation({
    mutationFn: ({
      file,
      lobName,
      llmEnabled,
    }: {
      file: File;
      lobName?: string;
      llmEnabled?: boolean;
    }) => api.analyze(file, lobName, llmEnabled),
  });
}
