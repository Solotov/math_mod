"""
settings.py - Configuración centralizada usando Pydantic Settings.
Carga variables de entorno desde .env y las valida.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class DataSettings(BaseSettings):
    data_dir: Path = Field(default="./data", env="DATA_DIR")
    cache_dir: Path = Field(default="./data/cache", env="CACHE_DIR")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    validation_split: float = Field(default=0.2, env="VALIDATION_SPLIT")
    shuffle: bool = Field(default=True, env="SHUFFLE")
    num_workers: int = Field(default=0, env="NUM_WORKERS")

class ModelSettings(BaseSettings):
    models_dir: Path = Field(default="./models", env="MODELS_DIR")
    backups_dir: Path = Field(default="./backups", env="BACKUPS_DIR")
    max_backups: int = Field(default=5, env="MAX_BACKUPS")
    default_epochs: int = Field(default=100, env="DEFAULT_EPOCHS")
    default_learning_rate: float = Field(default=0.001, env="DEFAULT_LEARNING_RATE")
    default_optimizer: str = Field(default="adam", env="DEFAULT_OPTIMIZER")
    default_loss: str = Field(default="mse", env="DEFAULT_LOSS")

class LoggingSettings(BaseSettings):
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_to_console: bool = Field(default=True, env="LOG_TO_CONSOLE")
    log_to_db: bool = Field(default=True, env="LOG_TO_DB")

class DatabaseSettings(BaseSettings):
    database_url: str = Field(default="sqlite:///./data/iadc.db", env="DATABASE_URL")

class Settings(BaseSettings):
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    logging: LoggingSettings = LoggingSettings()
    database: DatabaseSettings = DatabaseSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Singleton instance
settings = Settings()