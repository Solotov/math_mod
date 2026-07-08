#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/emed/proyects/python_playground/math_mod"

# -------------------------
# Directorios
# -------------------------
DIRS=(
    "$ROOT/config"
    "$ROOT/domain"
    "$ROOT/core"
    "$ROOT/infrastructure/database"
    "$ROOT/infrastructure/data"
    "$ROOT/infrastructure/ml"
    "$ROOT/infrastructure/logging"
    "$ROOT/services"
)

# -------------------------
# Archivos a crear
# (NO se crean si ya existen)
# -------------------------
FILES=(
    "$ROOT/config/__init__.py"
    "$ROOT/config/settings.py"
    "$ROOT/config/enums.py"

    "$ROOT/domain/__init__.py"
    "$ROOT/domain/events.py"
    "$ROOT/domain/exceptions.py"
    "$ROOT/domain/interfaces.py"

    "$ROOT/core/__init__.py"
    "$ROOT/core/event_bus.py"

    "$ROOT/infrastructure/database/__init__.py"
    "$ROOT/infrastructure/database/session.py"

    "$ROOT/infrastructure/data/__init__.py"
    "$ROOT/infrastructure/data/cache.py"
    "$ROOT/infrastructure/data/dataset_loader.py"

    "$ROOT/infrastructure/ml/__init__.py"
    "$ROOT/infrastructure/ml/network.py"
    "$ROOT/infrastructure/ml/trainer.py"
    "$ROOT/infrastructure/ml/predictor.py"
    "$ROOT/infrastructure/ml/model_manager.py"

    "$ROOT/infrastructure/logging/__init__.py"
    "$ROOT/infrastructure/logging/console_logger.py"
    "$ROOT/infrastructure/logging/db_logger.py"

    "$ROOT/services/__init__.py"
    "$ROOT/services/training_service.py"
    "$ROOT/services/prediction_service.py"

    "$ROOT/main.py"
)

echo "Creando directorios..."

for dir in "${DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        chmod 775 "$dir"
        echo "  [+] $dir"
    else
        echo "  [=] $dir (ya existe)"
    fi
done

echo
echo "Creando archivos..."

for file in "${FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        touch "$file"
        chmod 664 "$file"
        echo "  [+] $file"
    else
        echo "  [=] $file (ya existe)"
    fi
done

echo
echo "Aplicando permisos recursivos..."

find "$ROOT" -type d -exec chmod 775 {} \;
find "$ROOT" -type f -exec chmod 664 {} \;

echo
echo "Estructura creada correctamente."
echo
echo "NOTA:"
echo " - config.py existente NO se modifica."
echo " - models.py existente NO se modifica."
echo " - repository.py existente NO se modifica."
echo " - .env existente NO se modifica."