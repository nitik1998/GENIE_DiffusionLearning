"""
experiment_tracker.py — Lightweight Experiment Logging
=========================================================
Appends experiment results to a shared JSON log and human-readable
Markdown table. No external dependencies beyond Python stdlib.

Usage (from any task script):
    from src.experiment_tracker import log_experiment
    log_experiment(
        task="task1",
        exp_name="baseline",
        params={"epochs": 30, "lr": 1e-3, "batch_size": 64},
        metrics={"psnr_db": 34.82, "mse_overall": 0.0003},
        status="SUCCESS",
    )
"""

import json
import os
import csv
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import OUTPUT_DIR


def log_experiment(
    task: str,
    exp_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    status: str = "SUCCESS",
    notes: Optional[str] = None,
) -> None:
    """
    Append one experiment record to the shared log files.

    Creates/appends to:
        outputs/experiments_log.json  — machine-readable full log
        outputs/experiments_log.md    — human-readable Markdown table
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "experiment": exp_name,
        "status": status,
        "params": params,
        "metrics": metrics,
    }
    if notes:
        record["notes"] = notes

    # ── JSON log ──
    json_path = os.path.join(OUTPUT_DIR, "experiments_log.json")
    entries = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                entries = []
    entries.append(record)
    with open(json_path, "w") as f:
        json.dump(entries, f, indent=2)

    # ── Markdown log ──
    md_path = os.path.join(OUTPUT_DIR, "experiments_log.md")
    is_new = not os.path.exists(md_path) or os.path.getsize(md_path) == 0

    with open(md_path, "a") as f:
        if is_new:
            f.write("# Experiment Log\n\n")
            f.write("| Timestamp | Task | Experiment | Status | Key Metrics | Notes |\n")
            f.write("|-----------|------|------------|--------|-------------|-------|\n")

        # Format key metrics compactly
        metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items())
        note_str = notes or ""
        ts = record["timestamp"][:19].replace("T", " ")
        f.write(f"| {ts} | {task} | {exp_name} | {status} | {metric_str} | {note_str} |\n")


def save_run_metrics(
    output_dir: str,
    metrics: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save metrics.json and metrics.csv inside an experiment output directory.

    Args:
        output_dir: Path like outputs/task1/baseline/
        metrics:    Dict of metric_name -> value
        params:     Optional dict of hyperparameters to include
    """
    os.makedirs(output_dir, exist_ok=True)

    payload = {"metrics": metrics}
    if params:
        payload["params"] = params
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()

    # JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # CSV (flat key-value)
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
