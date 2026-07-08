project/
├── config/
│   ├── __init__.py
│   ├── env_loader.py          # ← Cargar .env y validar
│   └── settings.py            # ← Configuración centralizada
├── core/
│   ├── __init__.py
│   ├── event_bus.py           # ← Sistema de eventos (pub/sub)
│   ├── exceptions.py          # ← Excepciones custom
│   └── types.py               # ← Type hints y enums
├── database/
│   ├── __init__.py
│   ├── models.py              # ✓ Ya existe
│   ├── repository.py          # ✓ Ya existe
│   ├── schema.sql             # ✓ Ya existe
│   └── session.py             # ← Gestión de sesiones
├── data/
│   ├── __init__.py
│   ├── dataset.py             # ← CSV, caching, validación
│   └── cache.py               # ← Cache intermedia (Dict/Redis optional)
├── ml/
│   ├── __init__.py
│   ├── network.py             # ← NeuralNetwork(nn.Module)
│   ├── trainer.py             # ← Entrenamiento + eventos
│   ├── predictor.py           # ← Predicciones
│   └── model_manager.py       # ← Serialización, backups
├── logging/
│   ├── __init__.py
│   ├── logger.py              # ← Logger genérico
│   ├── db_logger.py           # ← Persistencia en BD
│   └── console_logger.py      # ← Salida a consola
├── services/
│   ├── __init__.py
│   └── training_service.py    # ← Orquestación (usa todos los módulos)
├── main.py                    # ← Punto de entrada
├── .env.example
└── .env                       # ← Configuración