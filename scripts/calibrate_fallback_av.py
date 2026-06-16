from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from asl.infrastructure.hand_detection.mediapipe_detector import MediaPipeHandDetector
from asl.infrastructure.ml.pretrained.cv_heuristic_adapter import CVHeuristicAdapter


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def main() -> None:
    out_path = Path("configs/fallback_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = CVHeuristicAdapter()
    a_samples: list[tuple[int, float, float, float]] = []
    v_samples: list[tuple[int, float, float, float]] = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("Calibration started.")
    print("Hold A and press key 1 several times (10-20 samples).")
    print("Hold V and press key 2 several times (10-20 samples).")
    print("Press S to save calibration, Q to quit without saving.")

    with MediaPipeHandDetector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            landmarks = detector.detect(frame)
            crop = None
            features = None
            if landmarks is not None and landmarks.crop.size > 0:
                crop = landmarks.crop
                features = adapter.extract_features(crop)

            status = f"A samples: {len(a_samples)} | V samples: {len(v_samples)}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(
                frame,
                "1=record A  2=record V  S=save  Q=quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if features is not None:
                fingers, extent, solidity, aspect = features
                cv2.putText(
                    frame,
                    f"fingers={fingers} extent={extent:.3f} solidity={solidity:.3f} aspect={aspect:.3f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Fallback A/V Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Calibration cancelled.")
                break
            if key == ord("1") and features is not None:
                a_samples.append(features)
                print(f"Recorded A sample #{len(a_samples)}")
            if key == ord("2") and features is not None:
                v_samples.append(features)
                print(f"Recorded V sample #{len(v_samples)}")
            if key == ord("s"):
                if len(a_samples) < 5 or len(v_samples) < 5:
                    print("Need at least 5 samples for each of A and V before saving.")
                    continue

                a_ext = [x[1] for x in a_samples]
                a_sol = [x[2] for x in a_samples]
                v_ext = [x[1] for x in v_samples]
                v_sol = [x[2] for x in v_samples]
                v_asp = [x[3] for x in v_samples]

                payload = {
                    "a_min_extent": _percentile(a_ext, 25),
                    "a_min_solidity": _percentile(a_sol, 25),
                    "v_max_extent": _percentile(v_ext, 75),
                    "v_max_solidity": _percentile(v_sol, 75),
                    "v_min_aspect": _percentile(v_asp, 10),
                    "v_max_aspect": _percentile(v_asp, 90),
                }
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Saved calibration to {out_path}")
                print(payload)
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
