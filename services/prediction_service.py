from infrastructure.ml.predictor import Predictor
from infrastructure.ml.model_manager import ModelManager
from core.event_bus import EventBus
from typing import Any

class PredictionService:
    def __init__(self, model_manager: ModelManager, event_bus: EventBus):
        self.predictor = Predictor(model_manager, event_bus)

    def predict(self, model_name: str, input_data: Any) -> Any:
        return self.predictor.predict(model_name, input_data)