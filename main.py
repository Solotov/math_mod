#!/usr/bin/env python
import sys
import logging
from pathlib import Path

from config import create_db_and_tables, engine
from config.settings import settings
from core.event_bus import EventBus
from infrastructure.data.dataset_loader import DatasetLoader
from infrastructure.data.cache import InMemoryCache
from infrastructure.ml.model_manager import ModelManager
from infrastructure.logging.console_logger import ConsoleLogger
from infrastructure.logging.db_logger import DbLogger
from infrastructure.database.session import get_session
from services.training_service import TrainingService
from services.prediction_service import PredictionService
from domain.events import Event

def setup_logging():
    logging.basicConfig(level=settings.logging.log_level)

def main():
    create_db_and_tables()
    setup_logging()

    event_bus = EventBus()

    console_logger = ConsoleLogger()
    event_bus.subscribe(Event, console_logger.handle)

    with get_session() as session:
        db_logger = DbLogger(session)
        event_bus.subscribe(Event, db_logger.handle)

    cache = InMemoryCache()
    dataset_loader = DatasetLoader(event_bus=event_bus, cache=cache)
    model_manager = ModelManager(event_bus=event_bus)

    training_service = TrainingService(
        data_provider=dataset_loader,
        model_loader=model_manager,
        event_bus=event_bus
    )

    prediction_service = PredictionService(model_manager, event_bus)

    csv_path = Path("data/sample.csv")
    if csv_path.exists():
        print("Entrenando modelo...")
        result = training_service.run_training(
            csv_path=csv_path,
            model_name="test_model",
            target_col="target",
            hyperparams={"epochs": 10, "hidden_sizes": [32, 16]}
        )
        print(f"Entrenamiento completado: {result}")
    else:
        print(f"Archivo CSV no encontrado: {csv_path}")

if __name__ == "__main__":
    main()