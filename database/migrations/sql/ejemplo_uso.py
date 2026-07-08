#!/usr/bin/env python3
"""
Ejemplo de uso de la base de datos IADC
Demuestra cómo insertar y consultar datos
"""

import sqlite3
from datetime import datetime
import json


def conectar_db(db_path="iadc.db"):
    """Establece conexión con la BD"""
    return sqlite3.connect(db_path)


def ejemplo_insertar_modelo(conn):
    """Ejemplo: Insertar un nuevo modelo"""
    print("\n📍 Insertando modelo...")
    
    cursor = conn.cursor()
    
    modelo = {
        'nombre': 'modelo_clasificador_v1',
        'epochs': 100,
        'optimizer': 'Adam',
        'loss': 'CrossEntropyLoss',
        'learning_rate': 0.001,
        'estado': 'entrenando'
    }
    
    cursor.execute("""
        INSERT INTO model (nombre, epochs, optimizer, loss, learning_rate, estado)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (modelo['nombre'], modelo['epochs'], modelo['optimizer'], 
          modelo['loss'], modelo['learning_rate'], modelo['estado']))
    
    conn.commit()
    modelo_id = cursor.lastrowid
    print(f"✓ Modelo insertado con ID: {modelo_id}")
    return modelo_id


def ejemplo_crear_sesion_entrenamiento(conn, modelo_id):
    """Ejemplo: Crear una sesión de entrenamiento"""
    print("\n📍 Creando sesión de entrenamiento...")
    
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO training_session (modelo_id, modo, estado, epochs)
        VALUES (?, ?, ?, ?)
    """, (modelo_id, 'distribuido', 'en_progreso', 100))
    
    conn.commit()
    session_id = cursor.lastrowid
    print(f"✓ Sesión de entrenamiento creada con ID: {session_id}")
    return session_id


def ejemplo_registrar_epoch(conn, session_id):
    """Ejemplo: Registrar métricas de un epoch"""
    print("\n📍 Registrando métricas de epoch...")
    
    cursor = conn.cursor()
    
    # Simulamos 5 epochs
    for epoch_num in range(1, 6):
        loss_value = 0.5 - (epoch_num * 0.08)  # Simular mejora
        
        cursor.execute("""
            INSERT INTO epoch (session_id, epoch, loss, tiempo, learning_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, epoch_num, loss_value, 45.2, 0.001))
        
        print(f"  ✓ Epoch {epoch_num} - Loss: {loss_value:.4f}")
    
    conn.commit()


def ejemplo_registrar_prediccion(conn, modelo_id):
    """Ejemplo: Registrar una predicción"""
    print("\n📍 Registrando predicción...")
    
    cursor = conn.cursor()
    
    entrada = {"imagen": "datos_base64...", "dimensiones": [224, 224]}
    salida = {"clase": "gato", "confianza": 0.98}
    
    cursor.execute("""
        INSERT INTO prediction (entrada, salida, modelo_id)
        VALUES (?, ?, ?)
    """, (json.dumps(entrada), json.dumps(salida), modelo_id))
    
    conn.commit()
    print(f"✓ Predicción registrada")


def ejemplo_registrar_evento(conn, modelo_id):
    """Ejemplo: Registrar un evento en el log"""
    print("\n📍 Registrando evento...")
    
    cursor = conn.cursor()
    
    eventos = [
        ('Modelo cargado', 'Modelo clasificador cargado en memoria'),
        ('Entrenamiento iniciado', 'Iniciando entrenamiento con 100 epochs'),
    ]
    
    for evento, descripcion in eventos:
        cursor.execute("""
            INSERT INTO event_log (evento, descripcion, modelo_id)
            VALUES (?, ?, ?)
        """, (evento, descripcion, modelo_id))
        print(f"  ✓ {evento}")
    
    conn.commit()


def ejemplo_consultas(conn):
    """Ejemplo: Realizar consultas"""
    print("\n" + "="*70)
    print("📊 CONSULTAS DE EJEMPLO")
    print("="*70)
    
    cursor = conn.cursor()
    
    # Consulta 1: Listar todos los modelos
    print("\n📌 Todos los modelos:")
    print("-" * 70)
    cursor.execute("SELECT id, nombre, optimizer, learning_rate, estado FROM model")
    for row in cursor.fetchall():
        modelo_id, nombre, optimizer, lr, estado = row
        print(f"  ID: {modelo_id} | {nombre} | Optimizer: {optimizer} | LR: {lr} | {estado}")
    
    # Consulta 2: Sesiones de entrenamiento
    print("\n📌 Sesiones de entrenamiento:")
    print("-" * 70)
    cursor.execute("""
        SELECT ts.id, m.nombre, ts.modo, ts.estado, ts.epochs, ts.fecha_inicio
        FROM training_session ts
        JOIN model m ON ts.modelo_id = m.id
        ORDER BY ts.fecha_inicio DESC
    """)
    for row in cursor.fetchall():
        session_id, modelo, modo, estado, epochs, fecha = row
        print(f"  ID: {session_id} | {modelo} | {modo} | {estado} | {epochs} epochs")
    
    # Consulta 3: Métricas por epoch
    print("\n📌 Métricas por epoch:")
    print("-" * 70)
    cursor.execute("""
        SELECT e.epoch, e.loss, e.tiempo, e.learning_rate
        FROM epoch e
        ORDER BY e.epoch
        LIMIT 10
    """)
    for row in cursor.fetchall():
        epoch_num, loss, tiempo, lr = row
        print(f"  Epoch {epoch_num} | Loss: {loss:.4f} | Tiempo: {tiempo}s | LR: {lr}")
    
    # Consulta 4: Predicciones realizadas
    print("\n📌 Predicciones realizadas:")
    print("-" * 70)
    cursor.execute("""
        SELECT p.id, m.nombre, p.fecha, p.salida
        FROM prediction p
        JOIN model m ON p.modelo_id = m.id
        ORDER BY p.fecha DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        pred_id, modelo, fecha, salida = row
        resultado = json.loads(salida)
        print(f"  ID: {pred_id} | {modelo} | {resultado}")
    
    # Consulta 5: Evento log
    print("\n📌 Registro de eventos (últimos 5):")
    print("-" * 70)
    cursor.execute("""
        SELECT fecha, evento, descripcion
        FROM event_log
        ORDER BY fecha DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        fecha, evento, desc = row
        print(f"  {fecha} | {evento} | {desc}")


def main():
    """Función principal"""
    print("="*70)
    print("🔍 EJEMPLO DE USO - Base de Datos IADC")
    print("="*70)
    
    # Conectar a la BD
    conn = conectar_db("iadc.db")
    
    try:
        # Ejemplos de inserción
        modelo_id = ejemplo_insertar_modelo(conn)
        session_id = ejemplo_crear_sesion_entrenamiento(conn, modelo_id)
        ejemplo_registrar_epoch(conn, session_id)
        ejemplo_registrar_prediccion(conn, modelo_id)
        ejemplo_registrar_evento(conn, modelo_id)
        
        # Realizar consultas
        ejemplo_consultas(conn)
        
        print("\n" + "="*70)
        print("✅ Ejemplo completado exitosamente")
        print("="*70 + "\n")
        
    except sqlite3.Error as e:
        print(f"\n✗ Error en base de datos: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
