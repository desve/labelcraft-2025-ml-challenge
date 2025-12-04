# файл: src/experiment_utils.py

from typing import Optional
import pandas as pd
from src.experiment_manager import ExperimentManager

def get_exp_manager(experiments_root: str) -> ExperimentManager:
    return ExperimentManager(root_dir=experiments_root)

def show_last_experiments(
    experiments_root: str,
    n: int = 10,
) -> pd.DataFrame:
    em = get_exp_manager(experiments_root)
    summary = em.get_summary()
    df = summary[["experiment_id", "name", "macro_f1", "micro_f1"]].tail(n)
    display(df)
    return df

def show_top_nlp_experiments(
    experiments_root: str,
    task: str = "labelcraft_2025",
    metric: str = "macro_f1",
    top_n: int = 10,
    model_name_substr: str = "transformer_agent",
) -> pd.DataFrame:
    em = get_exp_manager(experiments_root)
    df_nlp = em.get_top_experiments(
        task=task,
        metric=metric,
        top_n=top_n,
        model_name_substr=model_name_substr,
    )
    display(
        df_nlp[[
            "experiment_id",
            "name",
            "model_name",
            "macro_f1",
            "micro_f1",
            "created_at",
        ]]
    )
    return df_nlp
