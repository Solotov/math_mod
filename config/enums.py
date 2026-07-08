from enum import Enum

class ModelStatus(str, Enum):
    CREATED = "created"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    SAVED = "saved"

class DataType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    PREDICT = "predict"

class EventType(str, Enum):
    DATASET_LOADED = "dataset_loaded"
    TRAINING_STARTED = "training_started"
    EPOCH_FINISHED = "epoch_finished"
    TRAINING_FINISHED = "training_finished"
    MODEL_SAVED = "model_saved"
    MODEL_LOADED = "model_loaded"
    BACKUP_CREATED = "backup_created"
    PREDICTION_COMPLETED = "prediction_completed"
    ERROR_OCCURRED = "error_occurred"