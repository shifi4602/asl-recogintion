"""
RealtimeUI — OpenCV webcam loop with MediaPipe hand overlay and live ASL prediction.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from loguru import logger


class RealtimeUI:

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name

    def run(self) -> None:
        from asl.config.container import container
        from asl.infrastructure.data.preprocessors.image_preprocessor import ImagePreprocessor
        from asl.infrastructure.hand_detection.mediapipe_detector import MediaPipeHandDetector

        inference = container.inference_service()
        preprocessor = ImagePreprocessor(target_size=224)
        recent_labels: deque[str] = deque(maxlen=5)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam.")
            return

        with MediaPipeHandDetector() as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                landmarks = detector.detect(frame)  # draws landmarks in-place

                if landmarks is not None and landmarks.crop.size > 0:
                    try:
                        processed = preprocessor.transform(landmarks.crop)
                        result = inference.predict(
                            _encode_frame(processed),
                            model_name=self._model_name,
                        )
                        recent_labels.append(result.label)
                        stable_label = _majority_label(recent_labels)
                        label = f"{stable_label} ({result.confidence_pct})"
                        cv2.putText(
                            frame, label,
                            (landmarks.x_min, landmarks.y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                        )
                        cv2.rectangle(
                            frame,
                            (landmarks.x_min, landmarks.y_min),
                            (landmarks.x_max, landmarks.y_max),
                            (0, 255, 0), 2,
                        )
                    except Exception as exc:
                        logger.debug("Inference skipped: {}", exc)

                cv2.imshow("ASL Recognition — press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def _encode_frame(processed: np.ndarray) -> bytes:
    """Convert a normalized float32 array back to JPEG bytes for the inference service."""
    uint8 = (processed * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", uint8)
    return buf.tobytes()


def _majority_label(labels: deque[str]) -> str:
    if not labels:
        return "nothing"
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)
