# Base de Datos IADC - Sistema de Tracking de Modelos ML

Sistema de base de datos SQLite para gestionar el ciclo de vida completo de modelos de machine learning, incluyendo entrenamiento, validación, predicciones y auditoría.

## 📋 Archivos Incluidos

### 1. **schema.sql**
Script SQL que define el esquema completo de la base de datos.

**Tablas creadas:**
- `tipo` - Tipos de datos (train, validation, predict)
- `model` - Información de modelos ML
- `training_session` - Sesiones de entrenamiento
- `epoch` - Métricas por epoch
- `batch` - Métricas por batch
- `csv_file` - Registro de archivos CSV utilizados
- `prediction` - Predicciones realizadas
- `weights_update` - Actualizaciones de pesos
- `event_log` - Registro de eventos del sistema

### 2. **init_db.py**
Script Python 3 que ejecuta el esquema SQL e inicializa la base de datos.

## 🚀 Uso Rápido

### Opción 1: Crear la BD en el directorio actual
```bash
python3 init_db.py
```
Crea una base de datos llamada `iadc.db` en el directorio actual.

### Opción 2: Crear la BD en una ubicación específica
```bash
python3 init_db.py /ruta/deseada/iadc
```
Crea la base de datos en `/ruta/deseada/iadc.db`.

### Opción 3: Usar como módulo Python
```python
from init_db import DatabaseInitializer

# Crear inicializador
db_init = DatabaseInitializer("mi_base_datos")

# Inicializar
db_init.initialize(verbose=True)
```

## 📊 Estructura de Tablas

### **tipo**
Tabla de referencia con los tipos de datos soportados.

| Columna | Tipo | Notas |
|---------|------|-------|
| id | INTEGER | Clave primaria |
| nombre | TEXT | Valores: 'train', 'validation', 'predict' |

### **model**
Información principal de cada modelo ML.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| nombre | TEXT | Nombre único del modelo |
| fecha_creacion | TIMESTAMP | Fecha de creación (auto) |
| ultima_modificacion | TIMESTAMP | Última actualización (auto) |
| epochs | INTEGER | Número de epochs configurado |
| optimizer | TEXT | Optimizador utilizado (ej: Adam) |
| loss | TEXT | Función de pérdida (ej: MSE) |
| learning_rate | REAL | Tasa de aprendizaje |
| estado | TEXT | Estado actual (creado, entrenando, etc.) |

### **training_session**
Registro de cada sesión de entrenamiento.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| modelo_id | INTEGER | FK a model(id) |
| fecha_inicio | TIMESTAMP | Inicio del entrenamiento (auto) |
| fecha_fin | TIMESTAMP | Fin del entrenamiento |
| modo | TEXT | Modo de entrenamiento |
| estado | TEXT | Estado (iniciado, completado, error) |
| epochs | INTEGER | Epochs ejecutados |
| csv_usados | TEXT | JSON o lista de CSV utilizados |
| tiempo | REAL | Tiempo total en segundos |

### **epoch**
Métricas por epoch durante entrenamiento.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| session_id | INTEGER | FK a training_session(id) |
| epoch | INTEGER | Número de epoch |
| loss | REAL | Pérdida en este epoch |
| tiempo | REAL | Tiempo del epoch en segundos |
| learning_rate | REAL | Learning rate utilizado |

### **batch**
Métricas por batch dentro de un epoch.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| epoch_id | INTEGER | FK a epoch(id) |
| batch | INTEGER | Número de batch |
| loss | REAL | Pérdida del batch |
| muestras | INTEGER | Número de muestras |

### **csv_file**
Registro de archivos de datos utilizados.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| nombre | TEXT | Nombre del archivo |
| hash | TEXT | Hash SHA256 (único) para integridad |
| filas | INTEGER | Número de filas |
| fecha | TIMESTAMP | Fecha de registro (auto) |
| tipo_id | INTEGER | FK a tipo(id) |
| procesado | INTEGER | 0=no, 1=sí |

### **prediction**
Registro de predicciones realizadas.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| fecha | TIMESTAMP | Fecha de predicción (auto) |
| entrada | TEXT | Datos de entrada (JSON) |
| salida | TEXT | Resultado predicho (JSON) |
| modelo_id | INTEGER | FK a model(id) |

### **weights_update**
Seguimiento de actualizaciones de pesos.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| epoch_id | INTEGER | FK a epoch(id) |
| numero_actualizacion | INTEGER | Número secuencial |
| loss | REAL | Pérdida después de actualizar |

### **event_log**
Auditoría de eventos importantes del sistema.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | INTEGER | Clave primaria |
| fecha | TIMESTAMP | Momento del evento (auto) |
| evento | TEXT | Tipo: Modelo cargado, guardado, CSV incorrecto, etc. |
| descripcion | TEXT | Detalles adicionales |
| modelo_id | INTEGER | FK a model(id) (opcional) |

## 🔍 Consultas de Ejemplo

### Ver todos los modelos
```sql
SELECT * FROM model;
```

### Últimas 5 sesiones de entrenamiento
```sql
SELECT * FROM training_session 
ORDER BY fecha_inicio DESC 
LIMIT 5;
```

### Métricas de un modelo específico
```sql
SELECT e.epoch, e.loss, e.tiempo, e.learning_rate
FROM epoch e
JOIN training_session ts ON e.session_id = ts.id
JOIN model m ON ts.modelo_id = m.id
WHERE m.nombre = 'mi_modelo'
ORDER BY e.epoch;
```

### Predicciones realizadas hoy
```sql
SELECT * FROM prediction
WHERE DATE(fecha) = DATE('now')
ORDER BY fecha DESC;
```

### Eventos registrados
```sql
SELECT fecha, evento, descripcion 
FROM event_log
ORDER BY fecha DESC
LIMIT 20;
```

## ⚙️ Características

✅ **Relaciones Referenciadas**: Uso de Foreign Keys para integridad referencial
✅ **Índices Optimizados**: Índices en columnas frecuentemente consultadas
✅ **Timestamps Automáticos**: Fechas que se registran automáticamente
✅ **Valores por Defecto**: Estados y flags con valores sensatos
✅ **Unicidad Garantizada**: Nombres de modelos y hashes de archivos únicos

## 📝 Notas

- La base de datos usa **SQLite 3+**
- Todos los timestamps están en UTC
- La tabla `tipo` se pre-rellena con los valores: 'train', 'validation', 'predict'
- Se crean índices automáticamente para optimizar consultas comunes
- Es posible agregar más campos sin perder compatibilidad

## 🛠️ Mantenimiento

### Exportar esquema actual
```bash
sqlite3 iadc.db ".schema" > backup_schema.sql
```

### Hacer un backup completo
```bash
sqlite3 iadc.db ".dump" > iadc_backup.sql
```

### Restaurar desde backup
```bash
sqlite3 iadc_new.db < iadc_backup.sql
```

## 📄 Licencia

Libre para usar y modificar según necesidad.
