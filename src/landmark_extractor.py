from __future__ import annotations

from typing import Iterable


def _iter_landmarks(landmark_list: object | None) -> Iterable[object]:
    if landmark_list is None:
        return []
    return getattr(landmark_list, "landmark", [])


def extract_landmark_vector(landmark_list: object | None, include_visibility: bool = False) -> list[float]:
    values: list[float] = []
    for landmark in _iter_landmarks(landmark_list):
        values.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
        if include_visibility and hasattr(landmark, "visibility"):
            values.append(float(landmark.visibility))
    return values


def extract_holistic_features(result) -> dict[str, list[float]]:
    return {
        "left_hand": extract_landmark_vector(getattr(result, "left_hand_landmarks", None)),
        "right_hand": extract_landmark_vector(getattr(result, "right_hand_landmarks", None)),
        "pose": extract_landmark_vector(getattr(result, "pose_landmarks", None), include_visibility=True),
        "face": extract_landmark_vector(getattr(result, "face_landmarks", None)),
    }


def flatten_feature_groups(feature_groups: dict[str, list[float]]) -> list[float]:
    flattened: list[float] = []
    for values in feature_groups.values():
        flattened.extend(values)
    return flattened

