import hashlib
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from domain.exceptions import DatasetLoadError, ValidationError
from domain.events import DatasetLoadedEvent
from core.event_bus import EventBus
from infrastructure.data.cache import InMemoryCache
from config.settings import settings
import numpy as np

class DatasetLoader:
    def __init__(self, event_bus: Optional[EventBus] = None, cache: Optional[InMemoryCache] = None):
        self.event_bus = event_bus
        self.cache = cache or InMemoryCache()
        self.batch_size = settings.data.batch_size
        self.validation_split = settings.data.validation_split
        self.shuffle = settings.data.shuffle
        self.num_workers = settings.data.num_workers

    def load_dataset(self, file_path: Path, target_col: Optional[str] = None, 
                     feature_cols: Optional[List[str]] = None, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """Carga un CSV, lo convierte a tensores y devuelve train/val DataLoaders."""
        # Calcular hash del archivo para cache
        file_hash = self._file_hash(file_path)
        cache_key = f"{file_hash}_{self.batch_size}_{self.validation_split}_{self.shuffle}"

        if self.cache.has(cache_key):
            cached = self.cache.get(cache_key)
            return cached["train_loader"], cached["val_loader"]

        # Leer CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise DatasetLoadError(f"Error reading CSV: {e}")

        # Validar columnas
        if target_col is None:
            target_col = df.columns[-1]
            feature_cols = df.columns[:-1].tolist() if feature_cols is None else feature_cols
        else:
            if target_col not in df.columns:
                raise ValidationError(f"Target column '{target_col}' not found")
            if feature_cols is None:
                feature_cols = [col for col in df.columns if col != target_col]
            else:
                for col in feature_cols:
                    if col not in df.columns:
                        raise ValidationError(f"Feature column '{col}' not found")

        # Limpiar datos: manejar nulos
        df = df.dropna()
        if df.empty:
            raise ValidationError("DataFrame is empty after dropping nulls")

        # Separar X e y
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)

        # Convertir a tensores
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).reshape(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)

        # Dividir en train/val
        total = len(dataset)
        val_size = int(total * self.validation_split)
        train_size = total - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Guardar en cache
        self.cache.set(cache_key, {"train_loader": train_loader, "val_loader": val_loader})

        # Publicar evento
        if self.event_bus:
            self.event_bus.publish(DatasetLoadedEvent(
                file_path=str(file_path),
                num_samples=total,
                num_features=len(feature_cols),
                split={"train": train_size, "val": val_size}
            ))

        return train_loader, val_loader

    def _file_hash(self, file_path: Path) -> str:
        """Calcula hash MD5 del archivo."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()