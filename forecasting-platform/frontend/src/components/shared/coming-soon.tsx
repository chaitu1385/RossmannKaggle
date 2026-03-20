"use client";

import { Construction } from "lucide-react";

interface ComingSoonProps {
  feature: string;
  description?: string;
}

export function ComingSoon({ feature, description }: ComingSoonProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-muted-foreground/30 bg-muted/30 p-8 text-center">
      <Construction className="mb-3 h-10 w-10 text-muted-foreground/50" />
      <h3 className="text-sm font-semibold text-muted-foreground">{feature}</h3>
      <p className="mt-1 text-xs text-muted-foreground/70">
        {description || "Requires API endpoint — coming in a future release"}
      </p>
    </div>
  );
}
