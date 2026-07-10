import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Any, Callable
import time
from domain.events import TrainingStartedEvent, EpochFinishedEvent, TrainingFinishedEvent, ErrorOccurredEvent
from core.event_bus import EventBus
from config.settings import settings
import logging

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 scheduler: Optional[_LRScheduler] = None,
                 epochs: Optional[int] = None,
                 event_bus: Optional[EventBus] = None,
                 device: Optional[torch.device] = None,
                 early_stopping_patience: Optional[int] = None,
                 checkpoint_callback: Optional[Callable] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs or settings.model.default_epochs
        self.event_bus = event_bus
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_callback = checkpoint_callback
        self.logger = logging.getLogger(__name__)

        self.model.to(self.device)

    def train(self) -> Dict[str, Any]:
        """Ejecuta el bucle de entrenamiento."""
        start_time = time.time()
        best_val_loss = float("inf")
        patience_counter = 0

        if self.event_bus:
            self.event_bus.publish(TrainingStartedEvent(
                model_name=self.model.__class__.__name__,
                epochs=self.epochs,
                batch_size=self.train_loader.batch_size,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                optimizer=self.optimizer.__class__.__name__,
                loss=self.criterion.__class__.__name__
            ))

        for epoch in range(1, self.epochs + 1):
            try:
                train_loss = self._train_epoch(epoch)
                val_loss = self._validate_epoch()
            except Exception as e:
                if self.event_bus:
                    self.event_bus.publish(ErrorOccurredEvent(error=e, context={"epoch": epoch}))
                raise

            if self.early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

            if self.checkpoint_callback and epoch % 5 == 0:
                self.checkpoint_callback(self.model, epoch)

            if self.event_bus:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.event_bus.publish(EpochFinishedEvent(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr
                ))

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        total_time = time.time() - start_time

        if self.event_bus:
            self.event_bus.publish(TrainingFinishedEvent(
                model_name=self.model.__class__.__name__,
                final_train_loss=train_loss,
                final_val_loss=val_loss,
                total_epochs=epoch,
                total_time=total_time
            ))

        return {
            "epochs": epoch,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "total_time": total_time
        }

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X.size(0)
        return total_loss / len(self.train_loader.dataset)

    def _validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
        return total_loss / len(self.val_loader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
        return {"loss": total_loss / len(dataloader.dataset)}