"""Per-epoch metric logging to CSV (for easy Phase 7 plotting) and a final JSON history dump."""
import csv
import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "history.csv"
        self.rows = []
        self._csv_file = None
        self._csv_writer = None

    def log_epoch(self, row: dict) -> None:
        self.rows.append(row)
        if self._csv_writer is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def save_json(self, filename: str = "history.json") -> None:
        with open(self.out_dir / filename, "w") as f:
            json.dump(self.rows, f, indent=2)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
