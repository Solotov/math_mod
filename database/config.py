"""
config.py - Configuración central del proyecto IADC
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine as create_sqlmodel_engine

import database.models # noqa: F401

# Directorio del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(exist_ok=True)

# Configuración de base de datos SQLite
DATABASE_URL = f"sqlite:///{DB_DIR}/iadc.db"

# Engine para SQLModel (con opciones optimizadas para SQLite)
engine = create_sqlmodel_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,  # Cambiar a True para ver las queries SQL generadas
)


def create_db_and_tables():
    """Crea todas las tablas definidas en los modelos SQLModel"""
    SQLModel.metadata.create_all(engine, tables=list(SQLModel.metadata.tables.values())) 


def get_engine():
    """Retorna el engine de la base de datos"""
    return engine
