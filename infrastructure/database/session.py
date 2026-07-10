from sqlmodel import Session
from config import engine  # Reutilizamos el engine del config.py existente

def get_session() -> Session:
    """Context manager para sesiones de BD."""
    with Session(engine) as session:
        yield session