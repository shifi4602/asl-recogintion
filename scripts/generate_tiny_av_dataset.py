from pathlib import Path
import random
import shutil
import cv2
import numpy as np

ROOT = Path("data/split")
SPEC = {
    "train": 120,
    "val": 30,
    "test": 30,
}
CLASSES = ["A", "V"]


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)

    for split, count in SPEC.items():
        for cls in CLASSES:
            target = ROOT / split / cls
            target.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                bg = random.randint(0, 30)
                image = np.full((224, 224, 3), bg, dtype=np.uint8)

                font_scale = random.uniform(4.2, 5.2)
                thickness = random.randint(8, 12)
                text_size, _ = cv2.getTextSize(cls, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_w, text_h = text_size

                x = max(5, (224 - text_w) // 2 + random.randint(-10, 10))
                y = min(215, (224 + text_h) // 2 + random.randint(-10, 10))
                fg = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
                cv2.putText(
                    image,
                    cls,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    fg,
                    thickness,
                    cv2.LINE_AA,
                )

                if random.random() < 0.35:
                    image = cv2.GaussianBlur(image, (5, 5), 0)

                cv2.imwrite(str(target / f"{cls}_{i:04d}.png"), image)

    print(f"Generated tiny dataset at {ROOT.resolve()}")


if __name__ == "__main__":
    main()
