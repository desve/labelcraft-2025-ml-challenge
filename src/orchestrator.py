"""
Публичный Orchestrator для Label Craft 2025.

Содержит только базовые сценарии:
- baseline_only: TF-IDF + LinearSVC (или LR).
- nlp_light: один фиксированный запуск RuBERT-tiny.

Расширенные сценарии (rubert-tiny2, ансамбли, авто-анализ истории
экспериментов) реализуются в приватном модуле в LabelCraft_2025_private.
"""

from typing import List, Dict, Any

from src.paths import DATA_PATH, EXPERIMENTS_ROOT
from src.experiment_manager import ExperimentManager

# предполагаем, что эти функции уже есть в соответствующих файлах
from labelcraft_baseline_agent import run_tfidf_linearsvc  # нужно создать-обёртку вокруг P2
from labelcraft_nlp_agent import run_labelcraft_transformer


class Orchestrator:
    def __init__(self, experiments_root: str = EXPERIMENTS_ROOT):
        self.em = ExperimentManager(root_dir=experiments_root)

    # --- Сценарии запуска ---

    def run_baseline_only(self) -> Dict[str, Any]:
        """
        Запускает TF-IDF + LinearSVC как основной baseline.
        Ожидается, что run_tfidf_linearsvc возвращает dict с experiment_id и метриками.
        """
        result = run_tfidf_linearsvc(
            data_path=DATA_PATH,
            text_col="text_clean",
            target_col="cat_id",
            sample_size=40_000,
            min_samples_per_class=2,
            random_state=42,
        )
        return {"agent": "baseline_linearsvc", **result}

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
        return {"agent": "nlp_rubert_tiny_p1", **result}

    def run_nlp_modern(self, data_path_for_nlp: str) -> Dict[str, Any]:
        """
        Запускает RuBERT-tiny2 (P1''): конфиг как в эксперименте deccc7ce-....
        data_path_for_nlp можно передавать с TextAugment (textaug.parquet).
        """
        result = run_labelcraft_transformer(
            data_path=data_path_for_nlp,
            text_col="text_clean",
            target_col="cat_id",
            model_name="cointegrated/rubert-tiny2",
            sample_size=40_000,
            max_length=160,
            batch_size=8,
            num_epochs=2,
            learning_rate=2e-5,
            random_state=42,
            use_class_weights=False,
        )
        return {"agent": "nlp_rubert_tiny2_p1pp", **result}

    # --- Высокоуровневый интерфейс ---

    def run_scenario(self, name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Запускает один из предопределённых сценариев:
        - 'baseline_only'
        - 'nlp_light'
        - 'nlp_modern'
        """
        results: List[Dict[str, Any]] = []

        if name == "baseline_only":
            results.append(self.run_baseline_only())
        elif name == "nlp_light":
            results.append(self.run_nlp_light())
        elif name == "nlp_modern":
            data_path_for_nlp = kwargs.get("data_path_for_nlp", DATA_PATH)
            results.append(self.run_nlp_modern(data_path_for_nlp=data_path_for_nlp))
        else:
            raise ValueError(f"Unknown scenario: {name}")

        return results

    # --- Аналитика поверх ExperimentManager ---

    def summarize_core_experiments(self) -> None:
        """
        Печатает:
        - лучший baseline (по macro/micro F1);
        - лучший трансформер;
        - отношение micro F1 трансформера к baseline.
        """
        summary = self.em.get_summary()

        # baseline: LinearSVC
        df_base = summary[summary["name"].astype(str).str.contains("baseline_tfidf_linearsvc")]
        best_base = df_base.dropna(subset=["macro_f1", "micro_f1"]).sort_values(
            by="macro_f1", ascending=False
        ).head(1)

        # nlp: transformer_agent (все RuBERT-ы и tiny2)
        df_nlp = self.em.get_top_experiments(
            task="labelcraft_2025",
            metric="macro_f1",
            top_n=20,
            model_name_substr="transformer_agent",
        )

        best_nlp = df_nlp.head(1)

        print("=== Best baseline (LinearSVC) ===")
        display(best_base[["experiment_id", "macro_f1", "micro_f1"]])

        print("=== Best NLP (transformer_agent) ===")
        display(best_nlp[["experiment_id", "model_name", "macro_f1", "micro_f1"]])

        if not best_base.empty and not best_nlp.empty:
            base_micro = float(best_base["micro_f1"].iloc[0])
            nlp_micro = float(best_nlp["micro_f1"].iloc[0])
            ratio = nlp_micro / base_micro if base_micro > 0 else 0.0
            print(f"micro_F1(NLP) / micro_F1(baseline) = {ratio:.3f}")
