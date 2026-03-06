import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --------------------------
# Plotting Model Performance
# --------------------------

def load_model_perf_csv(project_root: str, extract_round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str) -> pd.DataFrame:
    """Load model performance metrics from CSV file."""
    file_path = f"{project_root}/results/extractions/{dataset}/{extract_round_time}-{model_abbr}-{prompt_type}-perf.csv"
    df = pd.read_csv(file_path)
    return df


def load_evaluation_csv(project_root: str, evaluation_round_mark: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str, eval_method: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    file_path = f"{project_root}/results/evaluations/{dataset}/{evaluation_round_mark}-{model_abbr}-{prompt_type}-{eval_method}.csv"
    df = pd.read_csv(file_path)
    return df

def load_evaluation_csv_for_prompt(project_root: str, evaluation_round_mark: str, dataset: str, model_abbr: str, model_name: str, prompt_strategy_type: str, eval_method: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    file_path = f"{project_root}/results/prompts/{dataset}/{evaluation_round_mark}-{model_abbr}-{prompt_strategy_type}-{eval_method}.csv"
    df = pd.read_csv(file_path)
    return df