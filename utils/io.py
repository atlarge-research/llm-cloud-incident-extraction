import os
import sys
import re
import time
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --------------------------
# Round Time Generation
# --------------------------

def generate_round_time() -> str:
    """Generate a unique round time identifier for experiments."""
    return datetime.now().strftime('%Y%m%dT%H%M%S')


def get_date_from_round_time(round_time: str) -> str:
    """Extract date from round_time string."""
    return round_time.split('T')[0]


# --------------------------
# Metadata Management
# --------------------------

def load_metadata_csv(metadata_type: str) -> pd.DataFrame:
    """Load existing metadata CSV file or create new one."""
    metadata_file = f"results/{metadata_type}-metadata.csv"
    
    if os.path.exists(metadata_file):
        return pd.read_csv(metadata_file)
    else:
        # Create new metadata file with appropriate columns
        if metadata_type == "extract":
            columns = ['round_time', 'date', 'dataset', 'api', 'model', 'prompt_type', 'input_file_name', 'output_file_name']
        elif metadata_type == "evaluate":
            columns = ['round_time', 'date', 'dataset', 'api', 'model', 'prompt_type', 'eval_fields', 'eval_methods', 'input_file_name', 'output_file_name']
        else:
            raise ValueError(f"Unknown metadata type: {metadata_type}")
        
        return pd.DataFrame(columns=columns)


def save_metadata_csv(df_metadata: pd.DataFrame, metadata_type: str) -> None:
    """Save metadata DataFrame to CSV file."""
    os.makedirs("results", exist_ok=True)
    metadata_file = f"results/{metadata_type}-metadata.csv"
    df_metadata.to_csv(metadata_file, index=False)
    print(f"[Saved] {metadata_type} metadata to: {metadata_file}")


def add_extraction_metadata(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str, 
                          input_file_name: str, output_file_name: str) -> None:
    """Add a new extraction run to the metadata."""
    df_metadata = load_metadata_csv("extract")
    
    new_row = {
        'round_time': round_time,
        'date': get_date_from_round_time(round_time),
        'dataset': dataset,
        'model_abbr': model_abbr,
        'model_name': model_name,
        'prompt_type': prompt_type,
        'input_file_name': input_file_name,
        'output_file_name': output_file_name
    }
    
    df_metadata = pd.concat([df_metadata, pd.DataFrame([new_row])], ignore_index=True)
    save_metadata_csv(df_metadata, "extract")


def add_evaluation_metadata(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str,
                          eval_fields: list, eval_methods: list, input_file_name: str, output_file_name: str) -> None:
    """Add a new evaluation run to the metadata."""
    df_metadata = load_metadata_csv("evaluate")
    
    new_row = {
        'round_time': round_time,
        'date': get_date_from_round_time(round_time),
        'dataset': dataset,
        'model_abbr': model_abbr,
        'model_name': model_name,
        'prompt_type': prompt_type,
        'eval_fields': ','.join(eval_fields),
        'eval_methods': ','.join(eval_methods),
        'input_file_name': input_file_name,
        'output_file_name': output_file_name
    }
    
    df_metadata = pd.concat([df_metadata, pd.DataFrame([new_row])], ignore_index=True)
    save_metadata_csv(df_metadata, "evaluate")


# --------------------------
# Input Handling
# --------------------------

def load_csv(file_path, file_name):
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_path = os.path.join(project_root, file_path, file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV file not found: {full_path}")
    
    df = pd.read_csv(full_path)
    return df


def load_extraction_jsonl(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str):
    """Load extraction results jsonl file."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, 'results', 'extractions', f'{dataset}/{round_time}-{model_abbr}-{prompt_type}-resp.jsonl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Extraction file not found: {file_path}")
    
    df_extract = pd.read_json(file_path, lines=True, convert_dates=False)
    df_extract = preprocess_eval_df(df_extract, mode="extraction", dataset=dataset)
    return df_extract


def load_extraction_jsonl_for_prompt(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_strategy_type: str):
    """Load prompt extraction results jsonl file."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, 'results', 'prompts', f'{dataset}/{round_time}-{model_abbr}-{prompt_strategy_type}-resp.jsonl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt extraction file not found: {file_path}")
    
    df_extract = pd.read_json(file_path, lines=True, convert_dates=False)
    df_extract = preprocess_eval_df(df_extract, mode="extraction", dataset=dataset)
    return df_extract


def load_extraction_for_analysis_jsonl(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str):
    """Load extraction results jsonl file."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, 'results', 'analysis', f'{dataset}/{round_time}-{model_abbr}-{prompt_type}-resp.jsonl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Extraction file not found: {file_path}")
    
    df_extract = pd.read_json(file_path, lines=True, convert_dates=False)
    df_extract = preprocess_eval_df(df_extract, mode="extraction", dataset=dataset)
    return df_extract


def load_label_csv(dataset: str):
    df_label = load_csv("data/4_label_data", f"{dataset}_label.csv")
    df_label = preprocess_eval_df(df_label, mode="label", dataset=dataset)
    return df_label


# --------------------------
# Output Handling
# --------------------------

def parse_response_json(response: str) -> dict:
    """Extract and parse the JSON part of LLM response."""
    match = re.search(r'{.*}', response, re.DOTALL)
    if not match:
        raise ValueError("[Error] No JSON object found in response.")
    response_json = json.loads(match.group().strip())
    # print(f"Parsed JSON: {response_json}")  # Debugging output
    return response_json


def save_response_jsonl(response_json: dict, round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str) -> str:
    """Save response to JSONL file using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "extractions", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'{round_time}-{model_abbr}-{prompt_type}-resp.jsonl'
    path = os.path.join(output_dir, output_filename)
    
    with open(path, 'a') as f:
        json.dump(response_json, f)
        f.write('\n')
    
    return output_filename


def save_response_jsonl_for_analysis(response_json: dict, round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str) -> str:
    """Save response to JSONL file using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "analysis", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'{round_time}-{model_abbr}-{prompt_type}-resp.jsonl'
    path = os.path.join(output_dir, output_filename)
    
    with open(path, 'a') as f:
        json.dump(response_json, f)
        f.write('\n')
    
    return output_filename


def save_response_jsonl_for_prompt(response_json: dict, round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_strategy_type: str) -> str:
    """Save response to JSONL file using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "prompts", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'{round_time}-{model_abbr}-{prompt_strategy_type}-resp.jsonl'
    path = os.path.join(output_dir, output_filename)

    with open(path, 'a') as f:
        json.dump(response_json, f)
        f.write('\n')
    
    return output_filename


def calculate_cost(model_abbr: str, model_name: str, token_input: int, token_output: int) -> tuple[float, float]:
    """Calculate input and output costs based on API and model pricing."""
    if model_name == "gpt-3.5-turbo":
        # Model: GPT-3.5-turbo, Price per 1M tokens: input $0.50, output $1.50
        # source: https://platform.openai.com/docs/pricing, date: 2025-07-28
        cost_input = token_input * 0.5 / 1000000
        cost_output = token_output * 1.5 / 1000000
    elif model_name == "gpt-4o":
        # Model: GPT-4o, Price per 1M tokens: input $2.50, output $10.00
        # source: https://platform.openai.com/docs/pricing
        cost_input = token_input * 2.5 / 1000000
        cost_output = token_output * 10.0 / 1000000

    elif model_name == "claude-3-5-haiku-20241022":
        # Model: Claude-3.5-haiku, Price per 1M tokens: input $0.80, output $4
        # source: https://www.anthropic.com/pricing#api
        cost_input = token_input * 0.80 / 1000000
        cost_output = token_output * 4.0 / 1000000
    elif model_name == "claude-sonnet-4-20250514":
        # Model: Claude-sonnet-4, Price per 1M tokens: input $3.0, output $15.0
        # source: https://www.anthropic.com/pricing#api
        cost_input = token_input * 3.0 / 1000000
        cost_output = token_output * 15.0 / 1000000
    # if model == "gemini-1.5-flash":
    #     # Model: Gemini-1.5-flash, Price per 1M tokens: input $0.075, output $0.30
    #     # source: https://ai.google.dev/gemini-api/docs/pricing
    #     cost_input = token_input * 0.075 / 1000000
    #     cost_output = token_output * 0.30 / 1000000
    elif model_name == "gemini-2.0-flash":
        # Model: Gemini-2.0-flash, Price per 1M tokens: input $0.10, output $0.40
        # source: https://ai.google.dev/gemini-api/docs/pricing
        cost_input = token_input * 0.10 / 1000000
        cost_output = token_output * 0.40 / 1000000
    elif model_name == "gemini-2.5-pro":
        # Model: Gemini-2.5-pro, Price per 1M tokens: input $1.25, output $10.00
        # source: https://ai.google.dev/gemini-api/docs/pricing
        cost_input = token_input * 1.25 / 1000000
        cost_output = token_output * 10.0 / 1000000

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return cost_input, cost_output


def save_model_perf_csv(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str, token_input: int, token_output: int, latency: float) -> str:
    """Save model performance metrics using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "extractions", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f'{round_time}-{model_abbr}-{prompt_type}-perf.csv'
    path = os.path.join(output_dir, output_filename)

    cost_input, cost_output = calculate_cost(model_abbr, model_name, token_input, token_output)

    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("token_input,cost_input(USD),token_output,cost_output(USD),latency(s)\n")
    with open(path, 'a') as f:
        f.write(f"{token_input},{cost_input},{token_output},{cost_output},{latency}\n")
    
    return output_filename


def save_model_perf_csv_for_analysis(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str, token_input: int, token_output: int, latency: float) -> str:
    """Save model performance metrics using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "analysis", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f'{round_time}-{model_abbr}-{prompt_type}-perf.csv'
    path = os.path.join(output_dir, output_filename)

    cost_input, cost_output = calculate_cost(model_abbr, model_name, token_input, token_output)

    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("token_input,cost_input(USD),token_output,cost_output(USD),latency(s)\n")
    with open(path, 'a') as f:
        f.write(f"{token_input},{cost_input},{token_output},{cost_output},{latency}\n")
    
    return output_filename


def save_model_perf_csv_for_prompt(round_time: str, dataset: str, model_abbr: str, model_name: str, prompt_strategy_type: str, token_input: int, token_output: int, latency: float) -> str:
    """Save model performance metrics using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "prompts", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f'{round_time}-{model_abbr}-{prompt_strategy_type}-perf.csv'
    path = os.path.join(output_dir, output_filename)

    cost_input, cost_output = calculate_cost(model_abbr, model_name, token_input, token_output)

    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("token_input,cost_input(USD),token_output,cost_output(USD),latency(s)\n")
    with open(path, 'a') as f:
        f.write(f"{token_input},{cost_input},{token_output},{cost_output},{latency}\n")

    return output_filename
    

def save_evaluation_csv(df, evaluation_round_mark: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str, eval_method: str) -> str:
    """Save evaluation results using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "evaluations", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{evaluation_round_mark}-{model_abbr}-{prompt_type}-{eval_method}.csv"
    file_path = os.path.join(output_dir, output_filename)
    df.to_csv(file_path, index=True, header=True)
    print(f"[Saved] Evaluation results to: {file_path}")
    return output_filename


def save_evaluation_csv_for_prompt(df, evaluation_round_mark: str, dataset: str, model_abbr: str, model_name: str, prompt_strategy_type: str, eval_method: str) -> str:
    """Save evaluation results using round_time and return the output filename."""
    # Get the project root directory (parent of utils directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "results", "prompts", dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{evaluation_round_mark}-{model_abbr}-{prompt_strategy_type}-{eval_method}.csv"
    file_path = os.path.join(output_dir, output_filename)
    df.to_csv(file_path, index=True, header=True)
    print(f"[Saved] Evaluation results to: {file_path}")
    return output_filename

# --------------------------
# Data Processing for Evaluation
# --------------------------

def preprocess_eval_df(df: pd.DataFrame, mode: str, dataset: str) -> pd.DataFrame:
    """Standardize the format of extracted or labeled dataframe."""
    df = df.copy().reset_index(drop=True)

    if mode == "label":
        if dataset == "aws":
            # Handle NaN values and convert datetime safely
            df['label_start_time'] = pd.to_datetime(df['label_start_time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M:%S')
            df['label_end_time'] = pd.to_datetime(df['label_end_time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M:%S')
        elif dataset == "gcp":
            df['label_start_time'] = pd.to_datetime(df['label_start_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
            df['label_end_time'] = pd.to_datetime(df['label_end_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
        label_columns = ["label_service_category", "label_user_symptom_category", "label_root_cause_category"]
        for col in label_columns:
            if col in df.columns:
                # Handle NaN values and convert to string before applying .str operations
                df[col] = df[col].fillna('').astype(str).str.upper()
        label_columns = ["label_service_name", "label_location", "label_user_symptom", "label_root_cause"]
        for col in label_columns:
            if col in df.columns:
                # Handle NaN values and convert to string before applying .str operations
                df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
    
    if mode == "extraction":
        if dataset == "gcp":
            df['start_time'] = pd.to_datetime(df['start_time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
            df['end_time'] = pd.to_datetime(df['end_time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
        if "user_symptom_category" in df.columns:
            df['user_symptom_category'] = df['user_symptom_category'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) and x is not None else x)
        ext_columns = ["service_name", "location", "user_symptom", "root_cause"]
        for col in ext_columns:
            if col in df.columns:
                # Handle NaN values and convert to string before applying .str operations
                df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
    
    return df
