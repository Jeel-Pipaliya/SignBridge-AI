from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
from time import strftime

from src.camera import CameraConfig, CameraStream
from src.landmark_extractor import extract_holistic_features, flatten_feature_groups
from src.mediapipe_detector import MediaPipeHolisticDetector


@dataclass(slots=True)
class CollectionSample:
    label: str
    features: list[float] = field(default_factory=list)


class SignDataCollector:
    def __init__(self, output_dir: str | Path = "data/landmarks") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_frame(self, frame, detector: MediaPipeHolisticDetector) -> CollectionSample:
        result = detector.process(frame)
        feature_groups = extract_holistic_features(result)
        return CollectionSample(label="unlabeled", features=flatten_feature_groups(feature_groups))

    def collect_result(self, result, label: str = "unlabeled") -> CollectionSample:
        feature_groups = extract_holistic_features(result)
        return CollectionSample(label=label, features=flatten_feature_groups(feature_groups))

    def save_csv(self, samples: list[CollectionSample], file_name: str) -> Path:
        file_path = self.output_dir / file_name
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for sample in samples:
                writer.writerow([sample.label, *sample.features])
        return file_path

    def save_labeled_sequence(self, label: str, samples: list[CollectionSample], file_name: str | None = None) -> Path:
        label_dir = self.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        output_name = file_name or f"sample_{strftime('%Y%m%d_%H%M%S')}.csv"
        return self.save_csv(samples, f"{label}/{output_name}")

    def build_camera(self, source: int | str = 0, width: int | None = None, height: int | None = None) -> CameraStream:
        return CameraStream(CameraConfig(source=source, width=width, height=height))

