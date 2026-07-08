BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "batch" (
	"epoch_id"	INTEGER NOT NULL,
	"batch"	INTEGER NOT NULL,
	"loss"	FLOAT,
	"muestras"	INTEGER,
	"id"	INTEGER NOT NULL,
	PRIMARY KEY("id"),
	FOREIGN KEY("epoch_id") REFERENCES "epoch"("id")
);
CREATE TABLE IF NOT EXISTS "csvfile" (
	"nombre"	VARCHAR NOT NULL,
	"hash"	VARCHAR NOT NULL,
	"filas"	INTEGER,
	"tipo_id"	INTEGER NOT NULL,
	"procesado"	INTEGER NOT NULL,
	"id"	INTEGER NOT NULL,
	"fecha"	DATETIME NOT NULL,
	PRIMARY KEY("id"),
	FOREIGN KEY("tipo_id") REFERENCES "tipo"("id")
);
CREATE TABLE IF NOT EXISTS "epoch" (
	"session_id"	INTEGER NOT NULL,
	"epoch"	INTEGER NOT NULL,
	"loss"	FLOAT,
	"tiempo"	FLOAT,
	"learning_rate"	FLOAT,
	"id"	INTEGER NOT NULL,
	PRIMARY KEY("id"),
	FOREIGN KEY("session_id") REFERENCES "trainingsession"("id")
);
CREATE TABLE IF NOT EXISTS "eventlog" (
	"evento"	VARCHAR NOT NULL,
	"descripcion"	VARCHAR,
	"modelo_id"	INTEGER,
	"id"	INTEGER NOT NULL,
	"fecha"	DATETIME NOT NULL,
	PRIMARY KEY("id"),
	FOREIGN KEY("modelo_id") REFERENCES "model"("id")
);
CREATE TABLE IF NOT EXISTS "model" (
	"nombre"	VARCHAR NOT NULL,
	"epochs"	INTEGER,
	"optimizer"	VARCHAR,
	"loss"	VARCHAR,
	"learning_rate"	FLOAT,
	"estado"	VARCHAR NOT NULL,
	"id"	INTEGER NOT NULL,
	"fecha_creacion"	DATETIME NOT NULL,
	"ultima_modificacion"	DATETIME NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "prediction" (
	"entrada"	VARCHAR NOT NULL,
	"salida"	VARCHAR NOT NULL,
	"modelo_id"	INTEGER NOT NULL,
	"id"	INTEGER NOT NULL,
	"fecha"	DATETIME NOT NULL,
	PRIMARY KEY("id"),
	FOREIGN KEY("modelo_id") REFERENCES "model"("id")
);
CREATE TABLE IF NOT EXISTS "tipo" (
	"nombre"	VARCHAR NOT NULL UNIQUE,
	"id"	INTEGER NOT NULL,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "trainingsession" (
	"modelo_id"	INTEGER NOT NULL,
	"modo"	VARCHAR,
	"estado"	VARCHAR NOT NULL,
	"epochs"	INTEGER,
	"csv_usados"	VARCHAR,
	"tiempo"	FLOAT,
	"id"	INTEGER NOT NULL,
	"fecha_inicio"	DATETIME NOT NULL,
	"fecha_fin"	DATETIME,
	PRIMARY KEY("id"),
	FOREIGN KEY("modelo_id") REFERENCES "model"("id")
);
CREATE TABLE IF NOT EXISTS "weightsupdate" (
	"epoch_id"	INTEGER NOT NULL,
	"numero_actualizacion"	INTEGER NOT NULL,
	"loss"	FLOAT,
	"id"	INTEGER NOT NULL,
	PRIMARY KEY("id"),
	FOREIGN KEY("epoch_id") REFERENCES "epoch"("id")
);
CREATE INDEX IF NOT EXISTS "ix_csvfile_fecha" ON "csvfile" (
	"fecha"
);
CREATE UNIQUE INDEX IF NOT EXISTS "ix_csvfile_hash" ON "csvfile" (
	"hash"
);
CREATE INDEX IF NOT EXISTS "ix_eventlog_fecha" ON "eventlog" (
	"fecha"
);
CREATE UNIQUE INDEX IF NOT EXISTS "ix_model_nombre" ON "model" (
	"nombre"
);
CREATE INDEX IF NOT EXISTS "ix_prediction_fecha" ON "prediction" (
	"fecha"
);
CREATE UNIQUE INDEX IF NOT EXISTS "ix_tipo_nombre" ON "tipo" (
	"nombre"
);
CREATE INDEX IF NOT EXISTS "ix_trainingsession_fecha_inicio" ON "trainingsession" (
	"fecha_inicio"
);
COMMIT;
