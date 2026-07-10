import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
from datetime import datetime
from domain.exceptions import ModelNotFoundError, ModelLoadError
from domain.events import ModelSavedEvent, ModelLoadedEvent, BackupCreatedEvent
from core.event_bus import EventBus
from config.settings import settings
from infrastructure.ml.network import NeuralNetwork

class ModelManager:
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.models_dir = Path(settings.model.models_dir)
        self.backups_dir = Path(settings.model.backups_dir)
        self.max_backups = settings.model.max_backups
        self.event_bus = event_bus
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, name: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        model_path = self.models_dir / f"{name}.pt"
        meta_path = self.models_dir / f"{name}.json"

        torch.save(model.state_dict(), model_path)

        if metadata is None:
            metadata = {}
        metadata["saved_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        if self.event_bus:
            self.event_bus.publish(ModelSavedEvent(
                model_name=name,
                file_path=str(model_path),
                metadata=metadata
            ))

        return model_path

    def load(self, name: str) -> torch.nn.Module:
        model_path = self.models_dir / f"{name}.pt"
        meta_path = self.models_dir / f"{name}.json"

        if not model_path.exists():
            raise ModelNotFoundError(f"Model {name} not found at {model_path}")

        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        input_size = metadata.get("input_size", 10)
        hidden_sizes = metadata.get("hidden_sizes", [20, 10])
        output_size = metadata.get("output_size", 1)

        model = NeuralNetwork(input_size, hidden_sizes, output_size)
        model.load_state_dict(torch.load(model_path))

        if self.event_bus:
            self.event_bus.publish(ModelLoadedEvent(
                model_name=name,
                file_path=str(model_path)
            ))

        return model

    def backup(self, model: torch.nn.Module, name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backups_dir / f"{name}_{timestamp}.pt"
        torch.save(model.state_dict(), backup_path)

        backups = sorted(self.backups_dir.glob(f"{name}_*.pt"))
        if len(backups) > self.max_backups:
            for old in backups[:-self.max_backups]:
                old.unlink()

        if self.event_bus:
            self.event_bus.publish(BackupCreatedEvent(
                model_name=name,
                backup_path=str(backup_path)
            ))

        return backup_path

    def list_models(self) -> List[str]:
        return [p.stem for p in self.models_dir.glob("*.pt")]