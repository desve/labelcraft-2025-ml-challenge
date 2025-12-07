# src/paths.py
"""
Централизованная конфигурация путей для Label Craft 2025.
Адаптируйте под своё пространство Google Drive.
"""
import os

# ================== КОНФИГУРАЦИЯ ПУТЕЙ ==================

# Корень клонированного репозитория на GitHub
PUBLIC_PROJECT_ROOT = "/content/drive/MyDrive/labelcraft-2025-ml-challenge"

# Корень проекта в Google Drive для Label Craft 2025
PROJECT_ROOT = "/content/drive/MyDrive/LabelCraft_2025"

# Приватная папка (для агентов, чекпоинтов, промежуточных данных)
PRIVATE_ROOT = "/content/drive/MyDrive/LabelCraft_2025_private"

# Папка для экспериментов
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")

# Путь к данным
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "labeled_train.parquet")

# Папка для чекпоинтов трансформеров (RuBERT и др.)
TRANSFORMER_OUTPUTS_ROOT = os.path.join(PRIVATE_ROOT, "transformer_checkpoints")

# Папка для приватных агентов
AGENTS_ROOT = os.path.join(PRIVATE_ROOT, "agents")

# ========================================================

def ensure_paths():
    """Создаёт все необходимые папки, если они не существуют."""
    dirs = [
        PROJECT_ROOT,
        PRIVATE_ROOT,
        EXPERIMENTS_ROOT,
        TRANSFORMER_OUTPUTS_ROOT,
        AGENTS_ROOT,
        os.path.join(PROJECT_ROOT, "data"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Все необходимые папки созданы/проверены")

def get_paths_summary():
    """Возвращает словарь со всеми основными путями."""
    return {
        "PUBLIC_PROJECT_ROOT": PUBLIC_PROJECT_ROOT,
        "PROJECT_ROOT": PROJECT_ROOT,
        "PRIVATE_ROOT": PRIVATE_ROOT,
        "EXPERIMENTS_ROOT": EXPERIMENTS_ROOT,
        "DATA_PATH": DATA_PATH,
        "TRANSFORMER_OUTPUTS_ROOT": TRANSFORMER_OUTPUTS_ROOT,
        "AGENTS_ROOT": AGENTS_ROOT,
    }

if __name__ == "__main__":
    ensure_paths()
    print("\n=== Текущие пути ===")
    for key, val in get_paths_summary().items():
        print(f"{key}: {val}")

