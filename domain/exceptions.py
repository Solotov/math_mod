class IADCError(Exception):
    """Base exception for IADC application."""
    pass

class ValidationError(IADCError):
    """Raised when data validation fails."""
    pass

class DatasetLoadError(IADCError):
    """Raised when dataset loading fails."""
    pass

class ModelNotFoundError(IADCError):
    """Raised when a model is not found."""
    pass

class ModelLoadError(IADCError):
    """Raised when model loading fails."""
    pass

class TrainingError(IADCError):
    """Raised during training."""
    pass

class PredictionError(IADCError):
    """Raised during prediction."""
    pass