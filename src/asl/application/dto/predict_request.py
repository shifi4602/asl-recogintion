from dataclasses import dataclass


@dataclass(frozen=True)
class PredictRequest:
    """
    image_bytes: raw image bytes (JPEG / PNG) decoded at the infrastructure layer.
    model_name:  name of the saved model to use (defaults to latest).
    top_k:       how many top predictions to return alongside the best one.
    """
    image_bytes: bytes
    model_name: str | None = None
    top_k: int = 3
