class ASLBaseException(Exception):
    """Base exception for all domain-level ASL errors."""


class ModelNotFoundError(ASLBaseException):
    """Raised when a saved model cannot be located."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Model '{name}' not found in the model repository.")


class DatasetNotReadyError(ASLBaseException):
    """Raised when a dataset split is requested before the data is downloaded/split."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"Dataset is not ready. {detail}".strip())


class InvalidImageError(ASLBaseException):
    """Raised when a supplied image cannot be decoded or has wrong shape."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"Invalid image data. {detail}".strip())


class TrainingError(ASLBaseException):
    """Raised when an error occurs during model training."""


class InferenceError(ASLBaseException):
    """Raised when an error occurs during sign prediction."""


class AuthenticationError(ASLBaseException):
    """Raised when authentication fails (invalid credentials or expired token)."""


class AuthorizationError(ASLBaseException):
    """Raised when an authenticated user lacks the required permissions."""
