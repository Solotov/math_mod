# -*- coding: utf-8 -*-
# File: main.py
# Description: Punto de entrada

from sqlmodel import Session, select
from sqlalchemy.orm import selectinload

from database.config import engine
from database.models import Model, ModelWithSessions



with Session(engine) as session:

    statement = (
        select(Model)
        .options(selectinload(Model.training_sessions))
        .where(Model.id == 1)
    )

    modelo = session.exec(statement).one_or_none()
    assert modelo is not None, "No se encontró el modelo con ID 1"
    resultado = ModelWithSessions.model_validate(modelo)

print(resultado)
