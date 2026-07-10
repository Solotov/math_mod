from domain.events import Event, EpochFinishedEvent, TrainingStartedEvent, TrainingFinishedEvent, ErrorOccurredEvent
import logging

class ConsoleLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("ConsoleLogger")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def handle(self, event: Event):
        if isinstance(event, TrainingStartedEvent):
            self.logger.info(f"Training started: {event.model_name}, epochs={event.epochs}, lr={event.learning_rate}")
        elif isinstance(event, EpochFinishedEvent):
            self.logger.info(f"Epoch {event.epoch} finished: train_loss={event.train_loss:.4f}, val_loss={event.val_loss:.4f}")
        elif isinstance(event, TrainingFinishedEvent):
            self.logger.info(f"Training finished: {event.model_name}, final_loss={event.final_train_loss:.4f}, time={event.total_time:.2f}s")
        elif isinstance(event, ErrorOccurredEvent):
            self.logger.error(f"Error: {event.error} - Context: {event.context}")
        else:
            self.logger.debug(f"Event received: {event}")