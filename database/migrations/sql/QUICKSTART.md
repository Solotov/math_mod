# 🚀 Guía de Inicio Rápido - IADC Database

## ¿Qué es IADC?

IADC (Intelligent AI Data Core) es un sistema de base de datos SQLite diseñado para gestionar y auditar el ciclo de vida completo de modelos de machine learning.

## ⚡ 5 Minutos para Empezar

### Paso 1: Inicializar la Base de Datos

```bash
python3 init_db.py
```

Esto crea un archivo `iadc.db` con todas las tablas necesarias.

**Resultado esperado:**
```
✅ Base de datos inicializada exitosamente
📁 Ubicación: /ruta/actual/iadc.db
```

### Paso 2: Ver el Ejemplo en Acción

```bash
python3 ejemplo_uso.py
```

Esto inserta datos de ejemplo y ejecuta consultas para demostrar cómo usar la BD.

### Paso 3: Comenzar a Integrar

Copia este código en tu proyecto:

```python
import sqlite3

# Conectar a la BD
conn = sqlite3.connect('iadc.db')
cursor = conn.cursor()

# Insertar un modelo
cursor.execute("""
    INSERT INTO model (nombre, epochs, optimizer, loss, learning_rate, estado)
    VALUES ('mi_modelo', 50, 'Adam', 'MSE', 0.001, 'creado')
""")
conn.commit()

# Consultar modelos
cursor.execute("SELECT * FROM model")
for modelo in cursor.fetchall():
    print(modelo)

conn.close()
```

## 📊 Las 9 Tablas Principales

| Tabla | Propósito |
|-------|-----------|
| `model` | Información del modelo ML |
| `training_session` | Sesiones de entrenamiento |
| `epoch` | Métricas por epoch |
| `batch` | Métricas por batch |
| `csv_file` | Registro de datos utilizados |
| `prediction` | Predicciones realizadas |
| `weights_update` | Cambios de pesos |
| `event_log` | Auditoría de eventos |
| `tipo` | Tipos de datos (train/val/predict) |

## 🔧 Casos de Uso Comunes

### Crear un Nuevo Modelo

```python
import sqlite3

conn = sqlite3.connect('iadc.db')
cursor = conn.cursor()

cursor.execute("""
    INSERT INTO model (nombre, epochs, optimizer, loss, learning_rate, estado)
    VALUES (?, ?, ?, ?, ?, ?)
""", ('modelo_detector_rostros', 100, 'SGD', 'CrossEntropy', 0.01, 'creado'))

conn.commit()
modelo_id = cursor.lastrowid
print(f"Modelo creado con ID: {modelo_id}")
conn.close()
```

### Registrar una Sesión de Entrenamiento

```python
cursor.execute("""
    INSERT INTO training_session (modelo_id, modo, estado, epochs)
    VALUES (?, ?, ?, ?)
""", (modelo_id, 'distribuido', 'en_progreso', 100))
conn.commit()
session_id = cursor.lastrowid
```

### Guardar Métricas de un Epoch

```python
cursor.execute("""
    INSERT INTO epoch (session_id, epoch, loss, tiempo, learning_rate)
    VALUES (?, ?, ?, ?, ?)
""", (session_id, 1, 0.5234, 120.45, 0.001))
conn.commit()
```

### Registrar una Predicción

```python
import json

entrada = {"datos": "..."}
salida = {"prediccion": "clase_A", "confianza": 0.95}

cursor.execute("""
    INSERT INTO prediction (entrada, salida, modelo_id)
    VALUES (?, ?, ?)
""", (json.dumps(entrada), json.dumps(salida), modelo_id))
conn.commit()
```

### Registrar Eventos

```python
cursor.execute("""
    INSERT INTO event_log (evento, descripcion, modelo_id)
    VALUES (?, ?, ?)
""", ('Modelo cargado', 'Modelo detector cargado en memoria', modelo_id))
conn.commit()
```

### Consultar Modelos con Mejor Desempeño

```python
cursor.execute("""
    SELECT m.nombre, MIN(e.loss) as mejor_loss, COUNT(e.id) as epochs
    FROM model m
    LEFT JOIN training_session ts ON m.id = ts.modelo_id
    LEFT JOIN epoch e ON ts.id = e.session_id
    GROUP BY m.id
    ORDER BY mejor_loss ASC
""")

for nombre, loss, epochs in cursor.fetchall():
    print(f"{nombre}: Loss={loss}, Epochs={epochs}")
```

## 📁 Estructura de Carpetas

```
proyecto/
├── iadc.db              # Base de datos (auto-generada)
├── schema.sql           # Definición de tablas
├── init_db.py           # Script inicializador
├── ejemplo_uso.py       # Ejemplo funcional
├── requirements.txt     # Dependencias
├── README.md            # Documentación completa
└── QUICKSTART.md        # Esta guía
```

## 🔒 Integridad y Seguridad

La base de datos incluye:

✅ **Foreign Keys**: Relaciones referenciadas entre tablas  
✅ **Índices**: Optimizados para consultas rápidas  
✅ **Restricciones UNIQUE**: Evita duplicados  
✅ **Timestamps**: Auditoría automática  
✅ **Hash de Archivos**: Integridad de datos  

## 💾 Backups

### Hacer Backup

```bash
# Backup completo
python3 -c "
import sqlite3
conn = sqlite3.connect('iadc.db')
with open('iadc_backup.sql', 'w') as f:
    for line in conn.iterdump():
        f.write(f'{line}\n')
conn.close()
print('✓ Backup guardado en iadc_backup.sql')
"
```

### Restaurar desde Backup

```bash
# Crear nueva BD desde backup
sqlite3 iadc_nuevo.db < iadc_backup.sql
```

## 🐛 Troubleshooting

### Error: "No such table"
Ejecuta `python3 init_db.py` para crear las tablas.

### Error: "database is locked"
Cierra todas las conexiones activas a la BD.

### Error: Foreign Key constraint failed
Asegúrate de que el modelo/sesión existe antes de insertarla como referencia.

## 📞 Soporte

- Ver `README.md` para documentación completa
- Ver `schema.sql` para definición exacta de tablas
- Ver `ejemplo_uso.py` para más ejemplos

## 🎯 Próximos Pasos

1. ✅ Inicializar la BD (`python3 init_db.py`)
2. ✅ Revisar el ejemplo (`python3 ejemplo_uso.py`)
3. ✅ Integrar en tu código
4. ✅ Registrar tus modelos y entrenamientos
5. ✅ Analizar métricas y eventos

¡Listo para comenzar! 🚀
