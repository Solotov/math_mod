-- Script de creación de tablas para base de datos iadc
-- SQLite

-- Tabla: tipo
-- Enum de tipos de datos de entrenamiento
CREATE TABLE IF NOT EXISTS tipo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT UNIQUE NOT NULL
);

-- Tabla: model
-- Almacena información de los modelos de ML
CREATE TABLE IF NOT EXISTS model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL UNIQUE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ultima_modificacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    epochs INTEGER,
    optimizer TEXT,
    loss TEXT,
    learning_rate REAL,
    estado TEXT DEFAULT 'creado'
);

-- Tabla: training_session
-- Registra sesiones de entrenamiento
CREATE TABLE IF NOT EXISTS training_session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    modelo_id INTEGER NOT NULL,
    fecha_inicio TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_fin TIMESTAMP,
    modo TEXT,
    estado TEXT DEFAULT 'iniciado',
    epochs INTEGER,
    csv_usados TEXT,
    tiempo REAL,
    FOREIGN KEY (modelo_id) REFERENCES model(id)
);

-- Tabla: epoch
-- Registra información por epoch durante entrenamiento
CREATE TABLE IF NOT EXISTS epoch (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    loss REAL,
    tiempo REAL,
    learning_rate REAL,
    FOREIGN KEY (session_id) REFERENCES training_session(id)
);

-- Tabla: batch
-- Registra información de cada batch
CREATE TABLE IF NOT EXISTS batch (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch_id INTEGER NOT NULL,
    batch INTEGER NOT NULL,
    loss REAL,
    muestras INTEGER,
    FOREIGN KEY (epoch_id) REFERENCES epoch(id)
);

-- Tabla: csv_file
-- Registro de archivos CSV utilizados
CREATE TABLE IF NOT EXISTS csv_file (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    hash TEXT UNIQUE NOT NULL,
    filas INTEGER,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tipo_id INTEGER NOT NULL,
    procesado INTEGER DEFAULT 0,
    FOREIGN KEY (tipo_id) REFERENCES tipo(id)
);

-- Tabla: prediction
-- Registra predicciones realizadas
CREATE TABLE IF NOT EXISTS prediction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    entrada TEXT NOT NULL,
    salida TEXT NOT NULL,
    modelo_id INTEGER NOT NULL,
    FOREIGN KEY (modelo_id) REFERENCES model(id)
);

-- Tabla: weights_update
-- Registra actualizaciones de pesos
CREATE TABLE IF NOT EXISTS weights_update (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch_id INTEGER NOT NULL,
    numero_actualizacion INTEGER NOT NULL,
    loss REAL,
    FOREIGN KEY (epoch_id) REFERENCES epoch(id)
);

-- Tabla: event_log
-- Registro de eventos del sistema
CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evento TEXT NOT NULL,
    descripcion TEXT,
    modelo_id INTEGER,
    FOREIGN KEY (modelo_id) REFERENCES model(id)
);

-- Insertar valores iniciales en tabla tipo
INSERT OR IGNORE INTO tipo (nombre) VALUES ('train');
INSERT OR IGNORE INTO tipo (nombre) VALUES ('validation');
INSERT OR IGNORE INTO tipo (nombre) VALUES ('predict');

-- Crear índices para optimizar consultas
CREATE INDEX IF NOT EXISTS idx_model_nombre ON model(nombre);
CREATE INDEX IF NOT EXISTS idx_training_session_modelo ON training_session(modelo_id);
CREATE INDEX IF NOT EXISTS idx_training_session_estado ON training_session(estado);
CREATE INDEX IF NOT EXISTS idx_epoch_session ON epoch(session_id);
CREATE INDEX IF NOT EXISTS idx_epoch_epoch_number ON epoch(epoch);
CREATE INDEX IF NOT EXISTS idx_batch_epoch ON batch(epoch_id);
CREATE INDEX IF NOT EXISTS idx_csv_file_hash ON csv_file(hash);
CREATE INDEX IF NOT EXISTS idx_csv_file_tipo ON csv_file(tipo_id);
CREATE INDEX IF NOT EXISTS idx_prediction_modelo ON prediction(modelo_id);
CREATE INDEX IF NOT EXISTS idx_prediction_fecha ON prediction(fecha);
CREATE INDEX IF NOT EXISTS idx_weights_update_epoch ON weights_update(epoch_id);
CREATE INDEX IF NOT EXISTS idx_event_log_modelo ON event_log(modelo_id);
CREATE INDEX IF NOT EXISTS idx_event_log_fecha ON event_log(fecha);
