import os

# 1. Идентификатор проекта
PROJECT_NAME = "labelcraft-2025-ml-challenge"

# 2. Открытый репозиторий (GitHub / локальный клон в Colab)
GITHUB_ROOT = "/content/drive/MyDrive/labelcraft-2025-ml-challenge"

# 3. Основной рабочий корень в Google Drive (Colab)
GDRIVE_ROOT = "/content/drive/MyDrive"

# 4. Папка проекта в Google Drive (для общедоступных артефактов)
PROJECT_ROOT = os.path.join(GDRIVE_ROOT, "LabelCraft_2025")

# 5. Приватный корень (можно перенести на Яндекс.Диск, достаточно сменить одну строку)
PRIVATE_ROOT = os.path.join(GDRIVE_ROOT, "LabelCraft_2025_private")
# Например, для Яндекс.Диска:
# PRIVATE_ROOT = "/content/drive/MyDrive/YandexDisk/LabelCraft_2025_private"

# 6. Подкаталоги внутри PRIVATE_ROOT
AGENTS_ROOT = os.path.join(PRIVATE_ROOT, "agents")
TRANSFORMER_OUTPUTS_ROOT = os.path.join(PRIVATE_ROOT, "labelcraft_transformer_outputs")

# 7. Данные
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "labeled_train.parquet")

# 8. Эксперименты (общий EXPERIMENTS_ROOT)
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
