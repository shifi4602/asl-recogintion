"""
Fallback ASL classifier backed by a public Hugging Face image model.
Used when no local .keras model exists yet.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from asl.domain.entities.sign_class import SignClass


class HFASLAdapter:
    """Expose a Keras-like .predict() API returning class probabilities."""

    def __init__(self, model_id: str = "dima806/asl_alphabet_image_detection") -> None:
        self._model_id = model_id
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._canonical_indices: list[int] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self._model_id)
        self._model.eval()
        self._canonical_indices = self._build_index_map()

    def _build_index_map(self) -> list[int]:
        if self._model is None:
            raise RuntimeError("Model must be loaded before building index map.")

        raw_id2label = getattr(self._model.config, "id2label", None)
        if not raw_id2label:
            raise RuntimeError("Pretrained model does not expose id2label mapping.")

        id2label = {int(k): str(v) for k, v in dict(raw_id2label).items()}
        label_to_index = {label: idx for idx, label in id2label.items()}

        canonical = SignClass.label_list()
        indices: list[int] = []
        for label in canonical:
            if label not in label_to_index:
                raise RuntimeError(f"Pretrained model missing expected label: {label}")
            indices.append(label_to_index[label])
        return indices

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._canonical_indices is not None

        if batch.ndim != 4:
            raise ValueError(f"Expected 4D batch [N,H,W,C], got shape {batch.shape}")

        images: list[Image.Image] = []
        for sample in batch:
            sample_u8 = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
            rgb = sample_u8[..., ::-1]
            images.append(Image.fromarray(rgb))

        inputs = self._processor(images=images, return_tensors="pt")
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1).cpu().numpy()

        probs = probs[:, self._canonical_indices]
        return probs
