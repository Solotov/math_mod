#!/usr/bin/env python3
"""
Script para inicializar la base de datos SQLite 'iadc'
Crea todas las tablas necesarias para el sistema de tracking de modelos ML.

Uso:
    python3 init_db.py              # Crea/inicializa iadc.db en el directorio actual
    python3 init_db.py /ruta/iadc   # Crea la DB en /ruta/iadc.db
"""

import sqlite3
import os
import sys
from pathlib import Path


class DatabaseInitializer:
    """Gestor para inicializar la base de datos iadc"""
    
    def __init__(self, db_path="iadc"):
        """
        Inicializa el gestor de BD.
        
        Args:
            db_path (str): Ruta de la base de datos (sin extensión .db)
        """
        # Asegurarse de que la ruta termina en .db
        if not db_path.endswith('.db'):
            db_path = f"{db_path}.db"
        
        self.db_path = db_path
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establece conexión con la base de datos"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            print(f"✓ Conectado a base de datos: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"✗ Error al conectar a la base de datos: {e}")
            return False
    
    def load_schema(self):
        """Carga el esquema SQL desde el archivo schema.sql"""
        schema_path = Path(__file__).parent / "schema.sql"
        
        if not schema_path.exists():
            print(f"✗ Archivo schema.sql no encontrado en {schema_path}")
            return None
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = f.read()
            print(f"✓ Esquema SQL cargado desde: {schema_path}")
            return schema
        except IOError as e:
            print(f"✗ Error al leer schema.sql: {e}")
            return None
    
    def execute_schema(self, schema):
        """Ejecuta el esquema SQL"""
        if not schema:
            return False
        
        try:
            self.cursor.executescript(schema)
            self.connection.commit()
            print("✓ Esquema ejecutado exitosamente")
            return True
        except sqlite3.Error as e:
            print(f"✗ Error al ejecutar esquema: {e}")
            self.connection.rollback()
            return False
    
    def verify_tables(self):
        """Verifica que todas las tablas se crearon correctamente"""
        expected_tables = [
            'tipo',
            'model',
            'training_session',
            'epoch',
            'batch',
            'csv_file',
            'prediction',
            'weights_update',
            'event_log'
        ]
        
        try:
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            existing_tables = [row[0] for row in self.cursor.fetchall()]
            
            print("\n📊 Verificación de tablas:")
            print("-" * 50)
            
            all_present = True
            for table in expected_tables:
                if table in existing_tables:
                    print(f"  ✓ {table}")
                else:
                    print(f"  ✗ {table} (FALTA)")
                    all_present = False
            
            print("-" * 50)
            
            if all_present:
                print(f"✓ Todas las {len(expected_tables)} tablas creadas correctamente")
                return True
            else:
                print("✗ Algunas tablas no fueron creadas")
                return False
                
        except sqlite3.Error as e:
            print(f"✗ Error al verificar tablas: {e}")
            return False
    
    def show_table_info(self):
        """Muestra información sobre las tablas creadas"""
        print("\n📋 Información detallada de tablas:")
        print("=" * 70)
        
        try:
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in self.cursor.fetchall()]
            
            for table in tables:
                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = self.cursor.fetchall()
                
                print(f"\n📌 Tabla: {table}")
                print("-" * 70)
                print(f"  {'Columna':<25} {'Tipo':<15} {'Nullable':<10} {'Default':<15}")
                print("-" * 70)
                
                for col_id, col_name, col_type, not_null, default_val, pk in columns:
                    nullable = "NO" if not_null else "SÍ"
                    default = default_val if default_val else "-"
                    print(f"  {col_name:<25} {col_type:<15} {nullable:<10} {default:<15}")
        
        except sqlite3.Error as e:
            print(f"✗ Error al obtener información: {e}")
    
    def close(self):
        """Cierra la conexión con la base de datos"""
        if self.connection:
            self.connection.close()
            print(f"\n✓ Conexión cerrada")
    
    def initialize(self, verbose=True):
        """Ejecuta el proceso completo de inicialización"""
        print("=" * 70)
        print("🚀 Inicializador de Base de Datos - IADC")
        print("=" * 70 + "\n")
        
        # Conectar
        if not self.connect():
            return False
        
        # Cargar esquema
        schema = self.load_schema()
        if not schema:
            self.close()
            return False
        
        # Ejecutar esquema
        if not self.execute_schema(schema):
            self.close()
            return False
        
        # Verificar tablas
        if not self.verify_tables():
            self.close()
            return False
        
        # Mostrar información detallada si se solicita
        if verbose:
            self.show_table_info()
        
        # Mostrar ruta final
        print("\n" + "=" * 70)
        print(f"✅ Base de datos inicializada exitosamente")
        print(f"📁 Ubicación: {os.path.abspath(self.db_path)}")
        print("=" * 70 + "\n")
        
        self.close()
        return True


def main():
    """Función principal"""
    # Procesar argumentos
    db_path = "iadc"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        # Si se pasa una ruta, extraer solo la ruta sin el nombre del archivo
        if '/' in db_path or '\\' in db_path:
            # Es una ruta completa o con directorio
            pass
        else:
            # Es solo un nombre
            db_path = db_path
    
    # Crear el directorio si no existe (en caso de ser una ruta)
    db_path = os.path.expanduser(db_path)
    db_dir = os.path.dirname(db_path) or "."
    
    if db_dir != "." and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
            print(f"📁 Directorio creado: {db_dir}\n")
        except OSError as e:
            print(f"✗ Error al crear directorio: {e}")
            sys.exit(1)
    
    # Inicializar
    initializer = DatabaseInitializer(db_path)
    success = initializer.initialize(verbose=True)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
