"""CSV experiment logging."""

import csv
import os
from typing import Dict, List
from config import ensure_dir

def append_metric_csv(path: str, row: Dict[str, object]) -> None:
    """
    每轮实时追加一行CSV
    """

    ensure_dir(os.path.dirname(path))

    file_exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        # 第一次写入增加header
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        # Make every completed round durable immediately. This is intentionally
        # done once per communication round, not once per local step.
        f.flush()
        os.fsync(f.fileno())




def write_metrics_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
