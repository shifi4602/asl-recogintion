"""Offline OpenCV heuristic ASL recognizer.

This fallback estimates a small subset of letters from hand contour geometry.
It is less accurate than a trained model but works without downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from asl.domain.entities.sign_class import SignClass


class CVHeuristicAdapter:
    """Expose a Keras-like .predict() API returning class probabilities."""

    def __init__(self, calibration_path: Path | None = None) -> None:
        self._calibration_path = calibration_path or Path("configs/fallback_calibration.json")
        self._thresholds = {
            "a_min_extent": 0.50,
            "a_min_solidity": 0.86,
            "v_max_extent": 0.52,
            "v_max_solidity": 0.84,
            "v_min_aspect": 0.45,
            "v_max_aspect": 1.35,
        }
        self._load_calibration()

    def _load_calibration(self) -> None:
        if not self._calibration_path.exists():
            return
        try:
            data = json.loads(self._calibration_path.read_text(encoding="utf-8"))
            for key in self._thresholds:
                if key in data:
                    self._thresholds[key] = float(data[key])
        except Exception:
            # Keep safe defaults if calibration file is malformed.
            return

    @staticmethod
    def _class_index(label: SignClass) -> int:
        return SignClass.label_list().index(label.value)

    @staticmethod
    def _distribution(label: SignClass) -> np.ndarray:
        n = SignClass.num_classes()
        probs = np.full(n, 0.0, dtype=np.float32)
        nothing_idx = CVHeuristicAdapter._class_index(SignClass.NOTHING)
        label_idx = CVHeuristicAdapter._class_index(label)

        if label is SignClass.NOTHING:
            probs.fill(0.35 / (n - 1))
            probs[nothing_idx] = 0.65
            return probs

        probs.fill(0.1 / (n - 2))
        probs[nothing_idx] = 0.15
        probs[label_idx] = 0.75
        return probs

    @staticmethod
    def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    @staticmethod
    def _count_fingers(mask: np.ndarray) -> int:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 3000:
            return 0

        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 4:
            return 0

        defects = cv2.convexityDefects(cnt, hull_indices)
        if defects is None:
            return 0

        fingers = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            if b == 0 or c == 0:
                continue

            angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2 * b * c)))
            if angle < 90 and d > 12000:
                fingers += 1

        # Defect counting is noisy on webcam frames, so keep this conservative.
        return min(fingers + 1, 5) if fingers > 0 else 0

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 2500:
            return None
        return cnt

    @staticmethod
    def _hand_shape_features(cnt: np.ndarray) -> tuple[float, float, float]:
        area = float(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = float(max(w * h, 1))
        extent = area / rect_area

        hull = cv2.convexHull(cnt)
        hull_area = float(max(cv2.contourArea(hull), 1.0))
        solidity = area / hull_area

        aspect = float(w) / float(max(h, 1))
        return extent, solidity, aspect

    def _detect_label(self, image_bgr: np.ndarray) -> SignClass:
        features = self.extract_features(image_bgr)
        if features is None:
            return SignClass.NOTHING

        fingers, extent, solidity, aspect = features

        # A (fist-like) tends to be compact and solid.
        if (
            fingers <= 1
            and extent > self._thresholds["a_min_extent"]
            and solidity > self._thresholds["a_min_solidity"]
        ):
            return SignClass.A

        # V (two spread fingers) should be less compact and less solid.
        if (
            fingers == 2
            and extent < self._thresholds["v_max_extent"]
            and solidity < self._thresholds["v_max_solidity"]
            and self._thresholds["v_min_aspect"] <= aspect <= self._thresholds["v_max_aspect"]
        ):
            return SignClass.V

        if fingers == 1:
            return SignClass.D
        if fingers == 3:
            return SignClass.W
        if fingers >= 4:
            return SignClass.B
        # Default to A for closed/ambiguous hand shapes to avoid constant V bias.
        return SignClass.A

    def extract_features(self, image_bgr: np.ndarray) -> tuple[int, float, float, float] | None:
        mask = self._skin_mask(image_bgr)
        cnt = self._largest_contour(mask)
        if cnt is None:
            return None
        fingers = self._count_fingers(mask)
        extent, solidity, aspect = self._hand_shape_features(cnt)
        return fingers, extent, solidity, aspect

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        if batch.ndim != 4:
            raise ValueError(f"Expected 4D batch [N,H,W,C], got shape {batch.shape}")

        outputs: list[np.ndarray] = []
        for sample in batch:
            sample_u8 = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
            label = self._detect_label(sample_u8)
            outputs.append(self._distribution(label))
        return np.stack(outputs, axis=0)
