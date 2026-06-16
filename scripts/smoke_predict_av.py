from pathlib import Path

import cv2
import numpy as np

from asl.domain.entities.sign_class import SignClass
from asl.infrastructure.persistence.tf_model_repository import TFModelRepository


def predict_image(model, image_path: Path) -> tuple[str, float]:
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (224, 224))
    x = image.astype("float32") / 255.0
    probs = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return SignClass.from_index(idx).value, float(probs[idx])


def main() -> None:
    repo = TFModelRepository(save_dir=Path("models"))
    model = repo.load("tiny_local_29")

    test_a = next(Path("data/split/test/A").glob("*.png"))
    test_v = next(Path("data/split/test/V").glob("*.png"))

    pred_a, conf_a = predict_image(model, test_a)
    pred_v, conf_v = predict_image(model, test_v)

    print(f"A sample: predicted={pred_a}, confidence={conf_a:.4f}, file={test_a.name}")
    print(f"V sample: predicted={pred_v}, confidence={conf_v:.4f}, file={test_v.name}")


if __name__ == "__main__":
    main()
