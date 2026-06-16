"""
MediaPipeHandDetector — IHandDetector implementation using Google MediaPipe Hands.
Detects a single hand, draws landmarks, and returns the bounding-box crop.
"""
from __future__ import annotations

import numpy as np

from asl.domain.interfaces.hand_detector import HandLandmarks, IHandDetector


class MediaPipeHandDetector(IHandDetector):

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        padding: int = 20,
    ) -> None:
        import mediapipe as mp

        self._padding = padding
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # IHandDetector
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> HandLandmarks | None:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_lm = results.multi_hand_landmarks[0]
        # Draw landmarks on the frame (in-place — caller decides whether to display)
        self._mp_draw.draw_landmarks(frame, hand_lm, self._mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]

        x_min = max(0, int(min(xs)) - self._padding)
        x_max = min(w, int(max(xs)) + self._padding)
        y_min = max(0, int(min(ys)) - self._padding)
        y_max = min(h, int(max(ys)) + self._padding)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        return HandLandmarks(crop=crop, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "MediaPipeHandDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
