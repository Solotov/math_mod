from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, List

@dataclass
class Event:
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""

@dataclass
class DatasetLoadedEvent(Event):
    file_path: str
    num_samples: int
    num_features: int
    split: Dict[str, int]
    event_type: str = "dataset_loaded"

@dataclass
class TrainingStartedEvent(Event):
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss: str
    event_type: str = "training_started"

@dataclass
class EpochFinishedEvent(Event):
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: float = 0.0
    event_type: str = "epoch_finished"

@dataclass
class TrainingFinishedEvent(Event):
    model_name: str
    final_train_loss: float
    final_val_loss: Optional[float] = None
    total_epochs: int
    total_time: float
    event_type: str = "training_finished"

@dataclass
class ModelSavedEvent(Event):
    model_name: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_type: str = "model_saved"

@dataclass
class ModelLoadedEvent(Event):
    model_name: str
    file_path: str
    event_type: str = "model_loaded"

@dataclass
class BackupCreatedEvent(Event):
    model_name: str
    backup_path: str
    event_type: str = "backup_created"

@dataclass
class PredictionCompletedEvent(Event):
    model_name: str
    input: Any
    output: Any
    event_type: str = "prediction_completed"

@dataclass
class ErrorOccurredEvent(Event):
    error: Exception
    context: Dict[str, Any] = field(default_factory=dict)
    event_type: str = "error_occurred"