"""
models.py - Modelos SQLModel que representan las tablas de la BD
Define tanto el modelo de BD como los esquemas de entrada/salida
"""
from datetime import datetime
from typing import Optional, List
from sqlmodel import Field, SQLModel, Relationship


# ============================================================================
# MODELO: Tipo (Enum de tipos de datos)
# ============================================================================
class TipoBase(SQLModel):
    """Base para modelo Tipo"""
    nombre: str = Field(index=True, unique=True)


class Tipo(TipoBase, table=True):
    """Tabla tipo - tipos de datos de entrenamiento (train, validation, predict)"""
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Relaciones
    csv_files: List["CsvFile"] = Relationship(back_populates="tipo")


class TipoRead(TipoBase):
    """Schema para lectura de Tipo"""
    id: int


# ============================================================================
# MODELO: Model (Modelos de ML)
# ============================================================================
class ModelBase(SQLModel):
    """Base para modelo Model"""
    nombre: str = Field(index=True, unique=True)
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    loss: Optional[str] = None
    learning_rate: Optional[float] = None
    estado: str = Field(default="creado")


class Model(ModelBase, table=True):
    """Tabla model - información de modelos de ML"""
    id: Optional[int] = Field(default=None, primary_key=True)
    fecha_creacion: datetime = Field(default_factory=datetime.utcnow)
    ultima_modificacion: datetime = Field(default_factory=datetime.utcnow)
    
    # Relaciones
    training_sessions: List["TrainingSession"] = Relationship(back_populates="modelo")
    predictions: List["Prediction"] = Relationship(back_populates="modelo")
    event_logs: List["EventLog"] = Relationship(back_populates="modelo")


class ModelCreate(ModelBase):
    """Schema para creación de Model"""
    pass


class ModelUpdate(SQLModel):
    """Schema para actualización de Model"""
    nombre: Optional[str] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    loss: Optional[str] = None
    learning_rate: Optional[float] = None
    estado: Optional[str] = None


class ModelRead(ModelBase):
    """Schema para lectura de Model"""
    id: int
    fecha_creacion: datetime
    ultima_modificacion: datetime


class ModelWithSessions(ModelRead):
    """Schema para lectura de Model con sus sesiones de entrenamiento"""
    training_sessions: List["TrainingSessionRead"] = []


# ============================================================================
# MODELO: TrainingSession (Sesiones de entrenamiento)
# ============================================================================
class TrainingSessionBase(SQLModel):
    """Base para modelo TrainingSession"""
    modelo_id: int = Field(foreign_key="model.id")
    modo: Optional[str] = None
    estado: str = Field(default="iniciado")
    epochs: Optional[int] = None
    csv_usados: Optional[str] = None  # JSON stringificado
    tiempo: Optional[float] = None


class TrainingSession(TrainingSessionBase, table=True):
    """Tabla training_session - registra sesiones de entrenamiento"""
    id: Optional[int] = Field(default=None, primary_key=True)
    fecha_inicio: datetime = Field(default_factory=datetime.utcnow, index=True)
    fecha_fin: Optional[datetime] = None
    
    # Relaciones
    modelo: Optional[Model] = Relationship(back_populates="training_sessions")
    epochsx: List["Epoch"] = Relationship(back_populates="training_session")


class TrainingSessionCreate(TrainingSessionBase):
    """Schema para creación de TrainingSession"""
    pass


class TrainingSessionUpdate(SQLModel):
    """Schema para actualización de TrainingSession"""
    estado: Optional[str] = None
    tiempo: Optional[float] = None
    csv_usados: Optional[str] = None


class TrainingSessionRead(TrainingSessionBase):
    """Schema para lectura de TrainingSession"""
    id: int
    fecha_inicio: datetime
    fecha_fin: Optional[datetime]


class TrainingSessionWithEpochs(TrainingSessionRead):
    """Schema con detalle de epochs"""
    epochsx: List["EpochRead"] = []


# ============================================================================
# MODELO: Epoch (Información por epoch)
# ============================================================================
class EpochBase(SQLModel):
    """Base para modelo Epoch"""
    session_id: int = Field(foreign_key="trainingsession.id")
    epoch: int
    loss: Optional[float] = None
    tiempo: Optional[float] = None
    learning_rate: Optional[float] = None


class Epoch(EpochBase, table=True):
    """Tabla epoch - información granular por epoch"""
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Relaciones
    training_session: Optional[TrainingSession] = Relationship(back_populates="epochs")
    batches: List["Batch"] = Relationship(back_populates="epoch")
    weights_updates: List["WeightsUpdate"] = Relationship(back_populates="epoch")


class EpochCreate(EpochBase):
    """Schema para creación de Epoch"""
    pass


class EpochUpdate(SQLModel):
    """Schema para actualización de Epoch"""
    loss: Optional[float] = None
    tiempo: Optional[float] = None
    learning_rate: Optional[float] = None


class EpochRead(EpochBase):
    """Schema para lectura de Epoch"""
    id: int


class EpochWithBatches(EpochRead):
    """Schema con detalle de batches"""
    batches: List["BatchRead"] = []


# ============================================================================
# MODELO: Batch (Información de cada batch)
# ============================================================================
class BatchBase(SQLModel):
    """Base para modelo Batch"""
    epoch_id: int = Field(foreign_key="epoch.id")
    batch: int
    loss: Optional[float] = None
    muestras: Optional[int] = None


class Batch(BatchBase, table=True):
    """Tabla batch - información de cada batch"""
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Relaciones
    epoch: Optional[Epoch] = Relationship(back_populates="batches")


class BatchCreate(BatchBase):
    """Schema para creación de Batch"""
    pass


class BatchRead(BatchBase):
    """Schema para lectura de Batch"""
    id: int


# ============================================================================
# MODELO: CsvFile (Archivos CSV utilizados)
# ============================================================================
class CsvFileBase(SQLModel):
    """Base para modelo CsvFile"""
    nombre: str
    hash: str = Field(unique=True, index=True)
    filas: Optional[int] = None
    tipo_id: int = Field(foreign_key="tipo.id")
    procesado: int = Field(default=0)


class CsvFile(CsvFileBase, table=True):
    """Tabla csv_file - registro de archivos CSV"""
    id: Optional[int] = Field(default=None, primary_key=True)
    fecha: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Relaciones
    tipo: Optional[Tipo] = Relationship(back_populates="csv_files")


class CsvFileCreate(CsvFileBase):
    """Schema para creación de CsvFile"""
    pass


class CsvFileUpdate(SQLModel):
    """Schema para actualización de CsvFile"""
    procesado: Optional[int] = None
    filas: Optional[int] = None


class CsvFileRead(CsvFileBase):
    """Schema para lectura de CsvFile"""
    id: int
    fecha: datetime


# ============================================================================
# MODELO: Prediction (Predicciones realizadas)
# ============================================================================
class PredictionBase(SQLModel):
    """Base para modelo Prediction"""
    entrada: str
    salida: str
    modelo_id: int = Field(foreign_key="model.id")


class Prediction(PredictionBase, table=True):
    """Tabla prediction - predicciones realizadas"""
    id: Optional[int] = Field(default=None, primary_key=True)
    fecha: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Relaciones
    modelo: Optional[Model] = Relationship(back_populates="predictions")


class PredictionCreate(PredictionBase):
    """Schema para creación de Prediction"""
    pass


class PredictionRead(PredictionBase):
    """Schema para lectura de Prediction"""
    id: int
    fecha: datetime


# ============================================================================
# MODELO: WeightsUpdate (Actualizaciones de pesos)
# ============================================================================
class WeightsUpdateBase(SQLModel):
    """Base para modelo WeightsUpdate"""
    epoch_id: int = Field(foreign_key="epoch.id")
    numero_actualizacion: int
    loss: Optional[float] = None


class WeightsUpdate(WeightsUpdateBase, table=True):
    """Tabla weights_update - actualizaciones de pesos durante el entrenamiento"""
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Relaciones
    epoch: Optional[Epoch] = Relationship(back_populates="weights_updates")


class WeightsUpdateCreate(WeightsUpdateBase):
    """Schema para creación de WeightsUpdate"""
    pass


class WeightsUpdateRead(WeightsUpdateBase):
    """Schema para lectura de WeightsUpdate"""
    id: int


# ============================================================================
# MODELO: EventLog (Registro de eventos)
# ============================================================================
class EventLogBase(SQLModel):
    """Base para modelo EventLog"""
    evento: str
    descripcion: Optional[str] = None
    modelo_id: Optional[int] = Field(default=None, foreign_key="model.id")


class EventLog(EventLogBase, table=True):
    """Tabla event_log - registro de eventos del sistema"""
    id: Optional[int] = Field(default=None, primary_key=True)
    fecha: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Relaciones
    modelo: Optional[Model] = Relationship(back_populates="event_logs")


class EventLogCreate(EventLogBase):
    """Schema para creación de EventLog"""
    pass


class EventLogRead(EventLogBase):
    """Schema para lectura de EventLog"""
    id: int
    fecha: datetime
