import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class ExperimentManager:
    """
    Унифицированный менеджер экспериментов для соревнований (например, Data Fusion 2025).

    Делает три базовые вещи:
    - регистрирует эксперимент (id, описание, конфиг, дата);
    - логирует метрики (macro_f1, micro_f1 и др.);
    - сохраняет всё в CSV/JSON в папке experiments/.
    """

    def __init__(self, root_dir: str = "experiments"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

        # основной CSV со сводкой по экспериментам
        self.summary_path = os.path.join(self.root_dir, "experiments_summary.csv")
        if not os.path.exists(self.summary_path):
            df = pd.DataFrame(
                columns=[
                    "experiment_id",
                    "name",
                    "description",
                    "created_at",
                    "config_path",
                    "metrics_path",
                    "macro_f1",
                    "micro_f1",
                ]
            )
            df.to_csv(self.summary_path, index=False)

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def _experiment_dir(self, experiment_id: str) -> str:
        path = os.path.join(self.root_dir, experiment_id)
        os.makedirs(path, exist_ok=True)
        return path

    def register_experiment(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Регистрирует новый эксперимент и сохраняет config в JSON.

        Возвращает experiment_id (строка).
        """
        experiment_id = str(uuid.uuid4())
        exp_dir = self._experiment_dir(experiment_id)

        # сохраняем конфиг в config.json (если он есть)
        config_path = None
        if config is not None:
            config_path = os.path.join(exp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

        # создаём пустой metrics.json (по желанию)
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

        # добавляем строку в общий summary CSV
        df = pd.read_csv(self.summary_path)
        new_row = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "created_at": self._now_iso(),
            "config_path": config_path,
            "metrics_path": metrics_path,
            "macro_f1": None,
            "micro_f1": None,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.summary_path, index=False)

        return experiment_id

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Логирует метрики для уже зарегистрированного эксперимента.

        Ожидает, что среди metrics могут быть 'macro_f1' и 'micro_f1'.
        """
        # читаем существующий metrics.json
        exp_dir = self._experiment_dir(experiment_id)
        metrics_path = os.path.join(exp_dir, "metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                current = json.load(f)
        else:
            current = {}

        current.update(metrics)
        current["_updated_at"] = self._now_iso()

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)

        # обновляем сводную таблицу (macro_f1, micro_f1, путь к metrics.json)
        df = pd.read_csv(self.summary_path)
        mask = df["experiment_id"] == experiment_id
        if not mask.any():
            raise ValueError(f"Experiment {experiment_id} not found in summary CSV")

        if "macro_f1" in metrics:
            df.loc[mask, "macro_f1"] = metrics["macro_f1"]
        if "micro_f1" in metrics:
            df.loc[mask, "micro_f1"] = metrics["micro_f1"]
        df.loc[mask, "metrics_path"] = metrics_path
        df.to_csv(self.summary_path, index=False)

    def get_summary(self) -> pd.DataFrame:
        """
        Возвращает DataFrame со сводкой по всем экспериментам.
        """
        return pd.read_csv(self.summary_path)

    def get_experiment_config(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает config эксперимента (dict) или None, если не найден.
        """
        df = pd.read_csv(self.summary_path)
        row = df[df["experiment_id"] == experiment_id]
        if row.empty:
            return None

        config_path = row["config_path"].iloc[0]
        if pd.isna(config_path) or not os.path.exists(config_path):
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_top_experiments(
        self,
        task: str,
        metric: str = "macro_f1",
        top_n: int = 10,
        model_name_substr: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Возвращает top-N экспериментов по заданной метрике
        (по убыванию), отфильтрованных по task и, опционально,
        по подстроке в имени модели.

        task            — значение поля 'task' в config.json;
        metric          — колонка в summary (например, 'macro_f1');
        top_n           — сколько строк вернуть;
        model_name_substr — если задано, берём только эксперименты,
                            у которых в config['model']['name'] есть эта подстрока.
        """
        df = self.get_summary()

        # фильтруем только те эксперименты, у которых есть путь к конфигу
        df = df.dropna(subset=["config_path"]).copy()

        tasks: list[str] = []
        model_names: list[Optional[str]] = []

        for _, row in df.iterrows():
            try:
                with open(row["config_path"], "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                tasks.append(None)
                model_names.append(None)
                continue

            tasks.append(cfg.get("task"))
            model_cfg = cfg.get("model", {})
            model_names.append(model_cfg.get("name"))

        df["task"] = tasks
        df["model_name"] = model_names

        # фильтр по задаче
        df = df[df["task"] == task]

        # фильтр по имени модели (если нужно)
        if model_name_substr is not None:
            df = df[df["model_name"].astype(str).str.contains(model_name_substr)]

        # убираем NaN в метрике и сортируем
        df = df.dropna(subset=[metric])
        df = df.sort_values(by=metric, ascending=False)

        return df.head(top_n)
