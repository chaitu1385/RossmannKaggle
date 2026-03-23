"use client";

interface ConfigViewerProps {
  yaml: string;
}

export function ConfigViewer({ yaml }: ConfigViewerProps) {
  return (
    <div className="overflow-x-auto rounded-lg border bg-muted/30 p-4">
      <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground">
        {yaml}
      </pre>
    </div>
  );
}
