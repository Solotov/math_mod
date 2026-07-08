# -*- coding: utf-8 -*-
# File: main.py
# Description: Punto de entrada


from sqlmodel import Session
from database.config import engine
from database.models import Model
from database.repository import Repository


with Session(engine) as session:
    iaModel = Repository[Model](session, Model)
    one = iaModel.get_by_id(1)
    if one:
        print(f"Modelo encontrado: {one.id} - {one.nombre} - {one.estado}")
    else:
        item = iaModel.create(Model(nombre="singular-prism",epochs=0,optimizer="adam",loss="categorical_crossentropy",learning_rate=0.001))
        print(f"Modelo creado: {item.id} - {item.nombre} - {item.estado}")
        iaModel.update(item.id, {"estado": "entrenando"})

print("Operación completada.")