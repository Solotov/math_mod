from typing import Protocol, Any, Dict, Optional, Tuple, List
from pathlib import Path
import torch
from torch.utils.data import DataLoader

class IDataProvider(Protocol):
    def load_dataset(self, file_path: Path, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """Returns train and validation DataLoaders."""
        ...

class IModelLoader(Protocol):
    def save(self, model: torch.nn.Module, name: str, metadata: Optional[Dict] = None) -> Path:
        ...
    def load(self, name: str) -> torch.nn.Module:
        ...
    def backup(self, model: torch.nn.Module, name: str) -> Path:
        ...
    def list_models(self) -> List[str]:
        ...

class ITrainer(Protocol):
    def train(self) -> None:
        ...
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        ...

class IPredictor(Protocol):
    def predict(self, input_data: Any) -> Any:
        ...

class ICache(Protocol):
    def get(self, key: str) -> Optional[Any]:
        ...
    def set(self, key: str, value: Any) -> None:
        ...
    def has(self, key: str) -> bool:
        ...