"use client";

import { AlertTriangle } from "lucide-react";

interface ErrorDisplayProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export function ErrorDisplay({ title = "Error", message, onRetry }: ErrorDisplayProps) {
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-6 text-center">
      <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-destructive" />
      <h3 className="font-semibold text-destructive">{title}</h3>
      <p className="mt-1 text-sm text-muted-foreground">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
}
