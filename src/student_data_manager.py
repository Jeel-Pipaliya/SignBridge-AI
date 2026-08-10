from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from statistics import mean


@dataclass(slots=True)
class StudentRecord:
    name: str
    age: int
    marks: float


class StudentDataManager:
    def __init__(self, file_path: str | Path = "data/students.csv") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_records(self, records: list[StudentRecord]) -> Path:
        with self.file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "age", "marks"])
            for record in records:
                writer.writerow([record.name, record.age, record.marks])
        return self.file_path

    def load_records(self) -> list[StudentRecord]:
        if not self.file_path.exists():
            return []

        records: list[StudentRecord] = []
        with self.file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    StudentRecord(
                        name=row["name"],
                        age=int(row["age"]),
                        marks=float(row["marks"]),
                    )
                )
        return records

    def summarize_marks(self, records: list[StudentRecord]) -> dict[str, float]:
        if not records:
            return {"average": 0.0, "maximum": 0.0, "minimum": 0.0}

        scores = [record.marks for record in records]
        return {
            "average": float(mean(scores)),
            "maximum": max(scores),
            "minimum": min(scores),
        }

