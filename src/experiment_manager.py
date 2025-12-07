# src/experiment_manager.py
"""
ExperimentManager для Label Craft 2025.
Логирует эксперименты в experiments_summary.csv с конфигами и метриками.
"""
import os
import json
import uuid
import pandas as pd
from datetime import datetime

class ExperimentManager:
    """Менеджер экспериментов с автоматической регистрацией в CSV."""
    
    def __init__(self, root_dir):
        """
        Args:
            root_dir: корневая папка для экспериментов (EXPERIMENTS_ROOT)
        """
        self.root_dir = root_dir
        self.summary_path = os.path.join(root_dir, "experiments_summary.csv")
        os.makedirs(root_dir, exist_ok=True)
        self._init_summary()
    
    def _init_summary(self):
        """Инициализирует experiments_summary.csv, если он не существует."""
        if not os.path.exists(self.summary_path):
            df = pd.DataFrame(columns=[
                "experiment_id", "name", "description", "created_at",
                "config_path", "metrics_path",
                "macro_f1", "micro_f1", "sample_size", "model_type",
                "backbone", "max_length", "batch_size", "num_epochs",
                "use_class_weights", "other_params"
            ])
            df.to_csv(self.summary_path, index=False)
            print(f"✓ Создан {self.summary_path}")
    
    def register_experiment(self, name, description, config, metrics):
        """
        Регистрирует эксперимент: создаёт папку, сохраняет config/metrics, добавляет в CSV.
        
        Args:
            name: краткое название эксперимента
            description: описание
            config: словарь с конфигурацией
            metrics: словарь с метриками (macro_f1, micro_f1 и др.)
        
        Returns:
            experiment_id (str)
        """
        exp_id = str(uuid.uuid4())
        exp_dir = os.path.join(self.root_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Пути к файлам
        config_path = os.path.join(exp_dir, "config.json")
        metrics_path = os.path.join(exp_dir, "metrics.json")
        
        # Сохранение config и metrics
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Формирование строки для CSV
        row = {
            "experiment_id": exp_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": config_path,
            "metrics_path": metrics_path,
            "macro_f1": metrics.get("macro_f1", None),
            "micro_f1": metrics.get("micro_f1", None),
            "sample_size": config.get("sample_size", None),
            "model_type": config.get("model_type", None),
            "backbone": config.get("backbone", None),
            "max_length": config.get("max_length", None),
            "batch_size": config.get("batch_size", None),
            "num_epochs": config.get("num_epochs", None),
            "use_class_weights": config.get("use_class_weights", None),
            "other_params": json.dumps(config.get("other_params", {}), ensure_ascii=False),
        }
        
        # Добавление в CSV
        df = pd.read_csv(self.summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.summary_path, index=False)
        
        print(f"✓ Эксперимент зарегистрирован: {exp_id} ({name})")
        return exp_id
    
    def load_summary(self):
        """Загружает experiments_summary.csv как DataFrame."""
        return pd.read_csv(self.summary_path)
    
    def get_best_experiments(self, metric="macro_f1", top_n=5):
        """Возвращает top_n экспериментов по указанной метрике."""
        df = self.load_summary()
        df = df.dropna(subset=[metric])
        df = df.sort_values(by=metric, ascending=False)
        return df.head(top_n)
