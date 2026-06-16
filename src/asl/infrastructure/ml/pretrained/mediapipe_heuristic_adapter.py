"""Offline heuristic ASL recognizer based on MediaPipe hand landmarks.

This fallback is intentionally simple and recognizes a subset of letters.
It is used only when no trained model artifact is available.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from asl.domain.entities.sign_class import SignClass


class MediaPipeHeuristicAdapter:
    """Expose a Keras-like .predict() API returning class probabilities."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
        )

    @staticmethod
    def _class_index(label: SignClass) -> int:
        return SignClass.label_list().index(label.value)

    @staticmethod
    def _to_distribution(label: SignClass) -> np.ndarray:
        n = SignClass.num_classes()
        probs = np.full(n, 0.0, dtype=np.float32)

        nothing_idx = MediaPipeHeuristicAdapter._class_index(SignClass.NOTHING)
        label_idx = MediaPipeHeuristicAdapter._class_index(label)

        if label is SignClass.NOTHING:
            probs.fill(0.3 / (n - 1))
            probs[nothing_idx] = 0.7
            return probs

        probs.fill(0.1 / (n - 2))
        probs[nothing_idx] = 0.15
        probs[label_idx] = 0.75
        return probs

    def _detect_label(self, image_bgr: np.ndarray) -> SignClass:
        rgb = image_bgr[..., ::-1]
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return SignClass.NOTHING

        lm = result.multi_hand_landmarks[0].landmark

        handedness = "Right"
        if result.multi_handedness and result.multi_handedness[0].classification:
            handedness = result.multi_handedness[0].classification[0].label

        # Finger extension heuristics.
        thumb_tip_x = lm[4].x
        thumb_ip_x = lm[3].x
        thumb_extended = thumb_tip_x < thumb_ip_x if handedness == "Right" else thumb_tip_x > thumb_ip_x

        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y
        ring_extended = lm[16].y < lm[14].y
        pinky_extended = lm[20].y < lm[18].y

        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return SignClass.A
        if not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return SignClass.B
        if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return SignClass.L
        if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return SignClass.V
        if not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
            return SignClass.W
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return SignClass.I
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return SignClass.Y

        return SignClass.NOTHING

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        if batch.ndim != 4:
            raise ValueError(f"Expected 4D batch [N,H,W,C], got shape {batch.shape}")

        out: list[np.ndarray] = []
        for sample in batch:
            sample_u8 = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
            label = self._detect_label(sample_u8)
            out.append(self._to_distribution(label))
        return np.stack(out, axis=0)
