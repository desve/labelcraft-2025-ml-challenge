from typing import List, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.paths import DATA_PATH, EXPERIMENTS_ROOT
from src.experiment_manager import ExperimentManager
from labelcraft_nlp_agent import run_labelcraft_transformer


class Orchestrator:
    """
    Публичный Orchestrator для Label Craft 2025.

    Содержит базовые сценарии:
    - baseline_only: TF-IDF + LinearSVC.
    - nlp_light: один фиксированный запуск RuBERT-tiny (P1).

    Расширенные сценарии (rubert-tiny2, ансамбли, авто-анализ истории)
    реализуются в приватном модуле в LabelCraft_2025_private.
    """

    def __init__(self, experiments_root: str = EXPERIMENTS_ROOT):
        self.em = ExperimentManager(root_dir=experiments_root)

    # === Сценарии запуска ===

    def run_baseline_only(self) -> Dict[str, Any]:
        """
        Запускает TF-IDF + LinearSVC как основной baseline.
        Реализовано как тонкий адаптер внутри оркестратора.
        """
        DATA_PATH_LOCAL = DATA_PATH
        TEXT_COL = "text_clean"
        TARGET_COL = "cat_id"
        SAMPLE_SIZE = 40_000
        MIN_SAMPLES_PER_CLASS = 2
        RANDOM_STATE = 42

        df = pd.read_parquet(DATA_PATH_LOCAL)

        if TEXT_COL not in df.columns:
            df["source_name"] = df["source_name"].fillna("")
            df["attributes"] = df["attributes"].fillna("")
            df[TEXT_COL] = df["source_name"] + " " + df["attributes"]

        df[TEXT_COL] = df[TEXT_COL].fillna("")

        if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

        value_counts = df[TARGET_COL].value_counts()
        keep_cats = value_counts[value_counts >= MIN_SAMPLES_PER_CLASS].index
        df = df[df[TARGET_COL].isin(keep_cats)].reset_index(drop=True)

        X_text = df[TEXT_COL].astype(str).values
        y_raw = df[TARGET_COL].values

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        X_train_text, X_valid_text, y_train, y_valid = train_test_split(
            X_text,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        tfidf = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
        )

        X_train = tfidf.fit_transform(X_train_text)
        X_valid = tfidf.transform(X_valid_text)

        model = LinearSVC(
            C=1.0,
            class_weight=None,
            max_iter=500,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        macro_f1 = f1_score(y_valid, y_pred, average="macro")
        micro_f1 = f1_score(y_valid, y_pred, average="micro")

        config = {
            "task": "labelcraft_2025",
            "dataset": {
                "train_path": DATA_PATH_LOCAL,
                "text_col": TEXT_COL,
                "target_col": TARGET_COL,
                "sample_size": SAMPLE_SIZE,
                "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
            },
            "cv": {
                "test_size": 0.2,
                "stratified": True,
                "random_state": RANDOM_STATE,
            },
            "model": {
                "name": "baseline_linearsvc",
                "vectorizer": {
                    "type": "tfidf",
                    "max_features": 20_000,
                    "ngram_range": (1, 2),
                },
                "classifier": {
                    "type": "LinearSVC",
                    "C": 1.0,
                    "class_weight": None,
                    "max_iter": 500,
                },
            },
            "seed": RANDOM_STATE,
        }

        experiment_id = self.em.register_experiment(
            name="baseline_tfidf_linearsvc_orchestrator",
            description="TF-IDF + LinearSVC baseline from Orchestrator",
            config=config,
        )

        self.em.log_metrics(
            experiment_id=experiment_id,
            metrics={
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1),
            },
        )

        return {
            "experiment_id": experiment_id,
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
        }

    def run_nlp_light(self) -> Dict[str, Any]:
        """
        Запускает RuBERT-tiny (P1: max_length=128, bs=8, 2 epochs, без class weights).
        """
        result = run_labelcraft_transformer(
            data_path=DATA_PATH,
            text_col="text_clean",
            target_col="cat_id",
            model_name="cointegrated/rubert-tiny",
            sample_size=40_000,
            max_length=128,
            batch_size=8,
            num_epochs=2,
            learning_rate=2e-5,
            random_state=42,
            use_class_weights=False,
        )
        return {"experiment_id": result["experiment_id"],
                "macro_f1": result["macro_f1"],
                "micro_f1": result["micro_f1"]}

    # === Высокоуровневый интерфейс ===

    def run_scenario(self, name: str) -> List[Dict[str, Any]]:
        """
        Запускает один из предопределённых сценариев:
        - 'baseline_only'
        - 'nlp_light'
        """
        results: List[Dict[str, Any]] = []

        if name == "baseline_only":
            results.append({"agent": "baseline_linearsvc", **self.run_baseline_only()})
        elif name == "nlp_light":
            results.append({"agent": "nlp_rubert_tiny_p1", **self.run_nlp_light()})
        else:
            raise ValueError(f"Unknown scenario: {name}")

        return results

    # === Аналитика поверх ExperimentManager (упрощённая) ===

    def summarize_core_experiments(self) -> None:
        """
        Печатает:
        - лучшие baseline-эксперименты (по имени baseline_linearsvc);
        - лучшие NLP-эксперименты (model_name содержит transformer_agent).
        """
        summary = self.em.get_summary()

        df_base = summary[
            summary["name"].astype(str).str.contains("baseline_tfidf_linearsvc", na=False)
        ].dropna(subset=["macro_f1", "micro_f1"])

        df_nlp = self.em.get_top_experiments(
            task="labelcraft_2025",
            metric="macro_f1",
            top_n=10,
            model_name_substr="transformer_agent",
        )

        print("=== Baseline (LinearSVC) ===")
        if not df_base.empty:
            display(df_base.sort_values(by="macro_f1", ascending=False).head(5))
        else:
            print("No baseline_linearsvc experiments found.")

        print("=== NLP (transformer_agent) ===")
        if not df_nlp.empty:
            display(df_nlp[["experiment_id", "model_name", "macro_f1", "micro_f1"]])
        else:
            print("No NLP experiments found.")
