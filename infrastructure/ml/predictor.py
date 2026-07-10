import torch
from typing import Any, Union
import numpy as np
from domain.exceptions import PredictionError
from domain.events import PredictionCompletedEvent
from core.event_bus import EventBus
from infrastructure.ml.model_manager import ModelManager

class Predictor:
    def __init__(self, model_manager: ModelManager, event_bus: Optional[EventBus] = None):
        self.model_manager = model_manager
        self.event_bus = event_bus

    def predict(self, model_name: str, input_data: Union[list, np.ndarray, torch.Tensor]) -> Any:
        """Realiza una predicción usando el modelo especificado."""
        try:
            model = self.model_manager.load(model_name)
        except Exception as e:
            raise PredictionError(f"Failed to load model {model_name}: {e}")

        model.eval()
        if isinstance(input_data, list):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data
        else:
            raise PredictionError("Input data must be list, ndarray, or Tensor")

        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        result = output.numpy()

        if self.event_bus:
            self.event_bus.publish(PredictionCompletedEvent(
                model_name=model_name,
                input=input_data,
                output=result.tolist()
            ))

        return result