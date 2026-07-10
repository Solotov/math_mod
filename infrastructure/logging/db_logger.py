from domain.events import Event, EpochFinishedEvent, TrainingStartedEvent, TrainingFinishedEvent, ModelSavedEvent, PredictionCompletedEvent, ErrorOccurredEvent, DatasetLoadedEvent
from database.models import TrainingSession, Epoch, Batch, EventLog, Model, Prediction
from database.repository import Repository
from sqlmodel import Session
from typing import Optional
import json

class DbLogger:
    def __init__(self, session: Session):
        self.session = session
        self.model_repo = Repository[Model](session, Model)
        self.training_session_repo = Repository[TrainingSession](session, TrainingSession)
        self.epoch_repo = Repository[Epoch](session, Epoch)
        self.batch_repo = Repository[Batch](session, Batch)
        self.event_log_repo = Repository[EventLog](session, EventLog)
        self.prediction_repo = Repository[Prediction](session, Prediction)
        self.current_session_id: Optional[int] = None

    def handle(self, event: Event):
        event_log = EventLog(
            evento=event.event_type,
            descripcion=str(event),
            modelo_id=None
        )
        self.event_log_repo.create(event_log)

        if isinstance(event, TrainingStartedEvent):
            session = TrainingSession(
                modelo_id=None,
                modo=event.event_type,
                estado="iniciado",
                epochs=event.epochs,
                csv_usados=None,
                tiempo=0
            )
            db_session = self.training_session_repo.create(session)
            self.current_session_id = db_session.id

        elif isinstance(event, EpochFinishedEvent):
            if self.current_session_id is not None:
                epoch = Epoch(
                    session_id=self.current_session_id,
                    epoch=event.epoch,
                    loss=event.train_loss,
                    tiempo=0,
                    learning_rate=event.learning_rate
                )
                self.epoch_repo.create(epoch)

        elif isinstance(event, TrainingFinishedEvent):
            if self.current_session_id is not None:
                session = self.training_session_repo.get_by_id(self.current_session_id)
                if session:
                    session.estado = "completado"
                    session.tiempo = event.total_time
                    self.training_session_repo.update(self.current_session_id, session)

        elif isinstance(event, PredictionCompletedEvent):
            pred = Prediction(
                entrada=json.dumps(event.input) if not isinstance(event.input, str) else event.input,
                salida=json.dumps(event.output) if not isinstance(event.output, str) else event.output,
                modelo_id=None
            )
            self.prediction_repo.create(pred)