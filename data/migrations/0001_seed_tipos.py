from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from database.config import engine
from database.models import Tipo

TIPOS = (
    "train", # Tipo para sesiones de entrenamiento
    "validation", # Tipo para sesiones de validación
    "test", # Tipo para sesiones de prueba
    "predict", # Tipo para sesiones de predicción
    "update",   # Tipo para sesiones de actualización de modelo
)


def upgrade():

    with Session(engine) as session:

        for nombre in TIPOS:
            try:
                session.add(Tipo(nombre=nombre))
                session.commit()
            except IntegrityError:
                print(f"⚠️ Tipo '{nombre}' ya existe. Se omite.")
                session.rollback()

if __name__ == "__main__":
    print("Ejecutando migración para insertar tipos de sesiones...")
    upgrade()