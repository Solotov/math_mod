from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.optim import Adam, SGD
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from domain.interfaces import IDataProvider, IModelLoader
from domain.events import ErrorOccurredEvent
from core.event_bus import EventBus
from infrastructure.data.dataset_loader import DatasetLoader
from infrastructure.ml.model_manager import ModelManager
from infrastructure.ml.trainer import Trainer
from infrastructure.ml.network import NeuralNetwork
from config.settings import settings
import logging

class TrainingService:
    def __init__(self,
                 data_provider: IDataProvider,
                 model_loader: IModelLoader,
                 event_bus: EventBus):
        self.data_provider = data_provider
        self.model_loader = model_loader
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

    def run_training(self,
                     csv_path: Path,
                     model_name: str,
                     target_col: Optional[str] = None,
                     feature_cols: Optional[list] = None,
                     hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            train_loader, val_loader = self.data_provider.load_dataset(csv_path, target_col=target_col, feature_cols=feature_cols)

            sample = next(iter(train_loader))
            input_size = sample[0].shape[1]

            try:
                model = self.model_loader.load(model_name)
            except:
                hidden_sizes = hyperparams.get("hidden_sizes", [64, 32]) if hyperparams else [64, 32]
                output_size = hyperparams.get("output_size", 1) if hyperparams else 1
                model = NeuralNetwork(input_size, hidden_sizes, output_size)
                model.initialize_weights()

            lr = hyperparams.get("learning_rate", settings.model.default_learning_rate) if hyperparams else settings.model.default_learning_rate
            optimizer_name = hyperparams.get("optimizer", settings.model.default_optimizer) if hyperparams else settings.model.default_optimizer
            loss_name = hyperparams.get("loss", settings.model.default_loss) if hyperparams else settings.model.default_loss
            epochs = hyperparams.get("epochs", settings.model.default_epochs) if hyperparams else settings.model.default_epochs

            if optimizer_name.lower() == "adam":
                optimizer = Adam(model.parameters(), lr=lr)
            elif optimizer_name.lower() == "sgd":
                optimizer = SGD(model.parameters(), lr=lr)
            else:
                optimizer = Adam(model.parameters(), lr=lr)

            if loss_name.lower() == "mse":
                criterion = MSELoss()
            elif loss_name.lower() == "crossentropy":
                criterion = CrossEntropyLoss()
            else:
                criterion = MSELoss()

            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                epochs=epochs,
                event_bus=self.event_bus,
                early_stopping_patience=hyperparams.get("early_stopping_patience", 10) if hyperparams else 10,
                checkpoint_callback=lambda m, e: self.model_loader.backup(m, model_name)
            )

            result = trainer.train()

            self.model_loader.save(model, model_name, metadata={
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
                "output_size": output_size,
                "epochs": epochs,
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "loss": loss_name
            })

            return result

        except Exception as e:
            self.event_bus.publish(ErrorOccurredEvent(error=e, context={"model_name": model_name}))
            raise