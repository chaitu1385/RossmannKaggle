"""
Data Lineage Tracker — trace data flow through the forecasting pipeline.

Records each processing step (ingest, cleanse, gap-fill, model fit,
forecast, evaluation) with inputs, outputs, and parent linkage.  Enables
end-to-end provenance from raw source to final forecast.

Integrates with ``PipelineContext.run_id`` from ``src.observability``.

Usage::

    from src.lineage import track, get_tracker

    track("ingest", "data-loader", inputs=["sales.csv"], outputs=["raw_sales"])
    track("cleanse", "gap-filler", inputs=["raw_sales"], outputs=["clean_sales"])

    tracker = get_tracker()
    chain = tracker.get_lineage_for_output("clean_sales")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class LineageTracker:
    """Track data lineage through pipeline steps.

    Parameters
    ----------
    output_dir : str
        Directory for the lineage log file.
    run_id : str, optional
        Pipeline run ID (from PipelineContext).
    """

    def __init__(self, output_dir: str = "working", run_id: Optional[str] = None):
        self._log_path = Path(output_dir) / "lineage.json"
        self._entries: list[dict] = []
        self._counter = 0
        self.run_id = run_id

    def record(
        self,
        step: str,
        agent: str,
        inputs: List[str],
        outputs: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a pipeline step.

        Args:
            step: Step name (e.g. "ingest", "cleanse", "model_fit").
            agent: Component that performed the step.
            inputs: List of input identifiers (file paths, table names).
            outputs: List of output identifiers.
            metadata: Optional dict of additional context.

        Returns:
            Entry ID (e.g. "lin_001").
        """
        self._counter += 1
        entry_id = f"lin_{self._counter:03d}"

        parent_ids = self._find_parents(inputs)

        entry = {
            "id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "agent": agent,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "metadata": metadata or {},
            "parent_ids": parent_ids,
        }

        self._entries.append(entry)
        return entry_id

    def get_lineage(self) -> list[dict]:
        """Return the full lineage log."""
        return list(self._entries)

    def get_lineage_for_output(self, output_path: str) -> list[dict]:
        """Walk the parent chain for a given output back to roots.

        Args:
            output_path: Output identifier to trace.

        Returns:
            List of entries from root to the target output.
        """
        # Find entry that produced this output
        target = None
        for entry in reversed(self._entries):
            if output_path in entry["outputs"]:
                target = entry
                break

        if target is None:
            return []

        # BFS walk of parents
        chain = [target]
        visited = {target["id"]}
        queue = list(target["parent_ids"])

        while queue:
            pid = queue.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            parent = self._get_entry_by_id(pid)
            if parent:
                chain.append(parent)
                queue.extend(parent["parent_ids"])

        chain.reverse()
        return chain

    def get_outputs_for_step(self, step: str) -> list[str]:
        """Get all outputs produced by a specific step.

        Args:
            step: Step name.

        Returns:
            List of output identifiers.
        """
        outputs = []
        for entry in self._entries:
            if entry["step"] == step:
                outputs.extend(entry["outputs"])
        return outputs

    def save(self) -> Path:
        """Write lineage log to JSON file.

        Returns:
            Path to the saved file.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, default=str)
        return self._log_path

    def load(self) -> None:
        """Load lineage log from JSON file."""
        if self._log_path.exists():
            with open(self._log_path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
            self._counter = len(self._entries)

    def clear(self) -> None:
        """Reset the lineage log."""
        self._entries = []
        self._counter = 0

    def summary(self) -> dict:
        """Return a summary of the lineage log.

        Returns:
            dict with ``total_entries``, ``steps``, ``first_timestamp``,
            ``last_timestamp``.
        """
        if not self._entries:
            return {"total_entries": 0, "steps": [], "first_timestamp": None,
                    "last_timestamp": None}

        steps = list(dict.fromkeys(e["step"] for e in self._entries))
        return {
            "total_entries": len(self._entries),
            "steps": steps,
            "first_timestamp": self._entries[0]["timestamp"],
            "last_timestamp": self._entries[-1]["timestamp"],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_parents(self, inputs: List[str]) -> list[str]:
        """Find entries whose outputs match the given inputs."""
        input_set = set(inputs)
        parents = []
        for entry in self._entries:
            if input_set & set(entry["outputs"]):
                parents.append(entry["id"])
        return parents

    def _get_entry_by_id(self, entry_id: str) -> Optional[dict]:
        """Look up an entry by ID."""
        for entry in self._entries:
            if entry["id"] == entry_id:
                return entry
        return None


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_singleton: Optional[LineageTracker] = None


def get_tracker(output_dir: str = "working", run_id: Optional[str] = None) -> LineageTracker:
    """Get or create the singleton LineageTracker.

    Args:
        output_dir: Directory for the lineage log.
        run_id: Pipeline run ID.

    Returns:
        LineageTracker instance.
    """
    global _singleton
    if _singleton is None:
        _singleton = LineageTracker(output_dir=output_dir, run_id=run_id)
    elif run_id and _singleton.run_id != run_id:
        _singleton.run_id = run_id
    return _singleton


def track(
    step: str,
    agent: str,
    inputs: List[str],
    outputs: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Record a step via the singleton tracker.

    Args:
        step: Step name.
        agent: Component name.
        inputs: Input identifiers.
        outputs: Output identifiers.
        metadata: Optional context.

    Returns:
        Entry ID.
    """
    return get_tracker().record(step, agent, inputs, outputs, metadata)


def reset_tracker() -> None:
    """Reset the singleton tracker."""
    global _singleton
    _singleton = None
