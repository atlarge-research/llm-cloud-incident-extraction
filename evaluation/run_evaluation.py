import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io import (
    load_extraction_jsonl, load_label_csv, save_evaluation_csv,
    generate_round_time, add_evaluation_metadata
)
from utils.evaluate import exact_match, token_level, bert_score


# --------------------------
# Main Runner
# --------------------------

def run_evaluation(round_mark: str, dataset: str, model_abbr: str, model_name: str, prompt_type: str, eval_fields: list, eval_methods: list):
    print(f"⚙️ Running Evaluation for: round_time={round_mark}, model_abbr={model_abbr}, model_name={model_name}, dataset={dataset}, prompt={prompt_type}, eval_methods={eval_methods}")

    # Generate unique round_time for this evaluation run
    evaluation_round_time = generate_round_time()
    print(f"🕐 Evaluation round time: {evaluation_round_time}")
    # evaluation_round_mark = "eval"
    evaluation_round_time = evaluation_round_time[:8]
    evaluation_round_mark = evaluation_round_time

    # Load input data
    input_file_name = f"{dataset}/{round_mark}-{model_abbr}-{model_name}-{prompt_type}-resp.jsonl"
    
    df_extract = load_extraction_jsonl(round_mark, dataset, model_abbr, model_name, prompt_type)
    df_label = load_label_csv(dataset)

    # Align DataFrame columns
    df_extract = df_extract[[col for col in eval_fields if col in df_extract.columns]]
    label_cols = ['label_' + col for col in eval_fields if 'label_' + col in df_label.columns]
    df_label = df_label[label_cols]

    # Check lengths match
    if len(df_extract) != len(df_label):
        raise ValueError(f"Length mismatch between extracted and label data: {len(df_extract)} != {len(df_label)}")

    # Track output files for metadata
    output_files = []

    for method in eval_methods:
        if method == "EM":
            result = exact_match(df_extract, df_label, eval_fields)
            output_file = save_evaluation_csv(result, evaluation_round_mark, dataset, model_abbr, model_name, prompt_type, "em")
            output_files.append(output_file)
        elif method == "TK":
            result = token_level(df_extract, df_label, eval_fields)
            output_file = save_evaluation_csv(result, evaluation_round_mark, dataset, model_abbr, model_name, prompt_type, "tk")
            output_files.append(output_file)
        elif method == "BS":
            result = bert_score(df_extract, df_label, eval_fields)
            output_file = save_evaluation_csv(result, evaluation_round_mark, dataset, model_abbr, model_name, prompt_type, "bs")
            output_files.append(output_file)
        else:
            print(f"[Warning] Unknown evaluation method: {method}")
    
    # Add metadata for this evaluation run
    if output_files:
        add_evaluation_metadata(
            round_time=evaluation_round_time,
            dataset=dataset,
            model_abbr=model_abbr,
            model_name=model_name,
            prompt_type=prompt_type,
            eval_fields=eval_fields,
            eval_methods=eval_methods,
            input_file_name=input_file_name,
            output_file_name=','.join(output_files)
        )
        print(f"📝 Added evaluation metadata for round: {evaluation_round_time}")
    
    print("✅ Evaluation completed.")
    print("="*100)


# --------------------------
# Entry Point
# --------------------------

if __name__ == "__main__":
    with open(f"config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        model_dict = config['models']
    ## for testing
    # model_dict = {
    # # # 'gpt-3.5': 'gpt-3.5-turbo',
    # # # 'gpt-4o': 'gpt-4o',
    # # # 'claude-3-5': 'claude-3-5-haiku-20241022',
    # # # 'claude-4': 'claude-sonnet-4-20250514',
    # # 'gemini-2.0': 'gemini-2.0-flash',
    # # # 'gemini-2.5': 'gemini-2.5-pro'
    #  }
    for model_abbr, model_name in model_dict.items():
        for dataset in ["gcp"]:
            for prompt_type in ['0', '1']:
                run_evaluation( 
                    extraction_round_mark="ext",  # Extraction mark 
                    dataset=dataset,
                    model_abbr=model_abbr,
                    model_name=model_name,
                    prompt_type=prompt_type,
                    eval_fields=[
                        'service_name', 'start_time', 'end_time', 'timezone', 
                        'service_category', 'user_symptom_category', 'user_symptom'
                    ],
                    eval_methods=["EM", "TK"],  
                )