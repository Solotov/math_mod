# -*- coding: utf-8 -*-
# File: main.py
# Description: Punto de entrada

# TODO: Implementar funcionalidad

import sqlite3

# Versión de SQLite que está usando Python
print(sqlite3.sqlite_version)

# También puedes obtener información más detallada
print(f"SQLite versión: {sqlite3.sqlite_version}")
print(f"Versión del módulo sqlite3: {sqlite3.version}") # type: ignore