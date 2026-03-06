import os
import sys
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io import (
    load_csv, parse_response_json, save_response_jsonl, save_model_perf_csv,
    generate_round_time, add_extraction_metadata
)
from utils.extract import call_gpt, call_claude, call_gemini


# --------------------------
# Prompt Generation
# --------------------------

def generate_prompt_from_row(template: str, row: pd.Series, operator: str) -> str:
    service_category_lst = ['COMPUTE', 'STORAGE', 'NETWORK', 'SECURITY', 'AI', 'MANAGEMENT', 'ANALYTICS', 'DATABASE', 'OTHERS', 'UNKNOWN']
    user_symp_lst = ['ERROR', 'UNAVIL', 'DELAY', 'DEPERF', 'OTHERS', 'UNKNOWN']
    user_symp_instruction = open('prompts/user_symp_instruction.txt').read()
    root_cause_lst = ['CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN']
    root_cause_instruction = open('prompts/root_cause_instruction.txt').read()
    if operator == 'aws':
        return template.format(
            input_title=row['service'],
            input_desc=row['description'],
            input_status=row['status'],
            service_category_lst=service_category_lst,
            user_symp_lst=user_symp_lst,
            user_symp_instruction=user_symp_instruction,
        )
    elif operator == 'azure':
        return template.format(
            input_desc=row['description'],
            service_category_lst=service_category_lst,
            user_symp_lst=user_symp_lst,
            user_symp_instruction=user_symp_instruction,
            root_cause_lst=root_cause_lst,
            root_cause_instruction=root_cause_instruction,
        )
    elif operator == 'gcp':
        return template.format(
            input_title=row['service'],
            input_desc=row['description'],
            input_ext_desc=row['external_description'],
            service_category_lst=service_category_lst,
            user_symp_lst=user_symp_lst,
            user_symp_instruction=user_symp_instruction,
        )


def get_prompt_path(dataset: str, prompt_type: str) -> str:
    """Build the prompt file path dynamically."""
    return f"prompts/prompt_{dataset}_{prompt_type}.txt"


# --------------------------
# Main Runner
# --------------------------

def run_extraction(dataset: str, model_abbr: str, model_name: str, prompt_type: str, round_mark: str = 'ext'):
    print(f"🔍 Running extraction: dataset={dataset}, model_abbr={model_abbr}, model_name={model_name}, prompt_type={prompt_type}")

    # Generate unique round_time for this extraction run
    round_time = generate_round_time()
    print(f"🕐 Extraction round time: {round_time}")

    # round_mark = round_time[:8]
    round_time = round_time[:8]

    # Load input data
    input_file_name = f"{dataset}_label.csv"
    df = load_csv("data/4_label_data", input_file_name)
    # df = df.head(3) # for testing
    template_path = get_prompt_path(dataset, prompt_type)
    template = open(template_path).read()

    # Initialize output filenames
    output_file_name = None
    perf_file_name = None

    for i, row in df.iterrows():
        try:
            prompt = generate_prompt_from_row(template, row, dataset)
            # print(i + "\n")

            if model_abbr == "gpt-3.5" or model_abbr == "gpt-4o":
                start_time = time.time()
                response, response_text = call_gpt(prompt, model_name)
                token_input = response.usage.prompt_tokens
                token_output = response.usage.completion_tokens
                latency = time.time() - start_time
            elif model_abbr == "claude-3-5" or model_abbr == "claude-4":
                start_time = time.time()
                response, response_text = call_claude(prompt, model_name)
                token_input = response.usage.input_tokens
                token_output = response.usage.output_tokens
                latency = time.time() - start_time
            elif model_abbr == "gemini-2.0" or model_abbr == "gemini-2.5":
                start_time = time.time()
                response, response_text = call_gemini(prompt, model_name)
                token_input = response.usage_metadata.prompt_token_count
                token_output = response.usage_metadata.candidates_token_count
                latency = time.time() - start_time
            else:
                print(f"[Warning] API/Model '{model_abbr}' is not implemented.")
                continue

            # Save performance metrics
            perf_file_name = save_model_perf_csv(
                round_time, dataset, model_abbr, model_name, prompt_type, 
                token_input, token_output, latency
            )

            response_json = parse_response_json(response_text)
            print(response_json)
            output_file_name = save_response_jsonl(response_json, round_time, dataset, model_abbr, model_name, prompt_type)
            print(f'[INFO] saved {i}')

        except Exception as e:
            print(f"[Error] Row {i} failed with error: {e}")
            # add a NaN row to the response_json
            if dataset == 'aws':
                response_json = {
                    'service_name': 'NaN',
                    'location': 'NaN',
                    'service_category': 'NaN',
                    'start_time': 'NaN',
                    'end_time': 'NaN',
                    'timezone': 'NaN',
                    'user_symptom': 'NaN',
                    'user_symptom_category': 'NaN',
                }
            elif dataset == 'azure':
                response_json = {
                    'service_name': 'NaN',
                    'location': 'NaN',
                    'service_category': 'NaN',
                    'start_time': 'NaN',
                    'end_time': 'NaN',
                    'timezone': 'NaN',
                    'user_symptom': 'NaN',
                    'user_symptom_category': 'NaN',
                    'root_cause': 'NaN',
                    'root_cause_category': 'NaN'
                }
            elif dataset == 'gcp':
                response_json = {
                    'service_name': 'NaN',
                    'service_category': 'NaN',
                    'start_time': 'NaN',
                    'end_time': 'NaN',
                    'timezone': 'NaN',
                    'user_symptom': 'NaN',
                    'user_symptom_category': 'NaN'
                }
            save_response_jsonl(response_json, round_time, dataset, model_abbr, model_name, prompt_type)
            print(f'[INFO] NaN row saved {i}')
            continue

    # Add metadata for this extraction run
    if output_file_name:
        add_extraction_metadata(
            round_time=round_time,
            dataset=dataset,
            model_abbr=model_abbr,
            model_name=model_name,
            prompt_type=prompt_type,
            input_file_name=input_file_name,
            output_file_name=output_file_name
        )
        print(f"📝 Added extraction metadata for round: {round_time}")

    print("✅ Extraction completed.")
    print(f"="*100)


# --------------------------
# Entry Point
# --------------------------

if __name__ == "__main__":
    run_extraction()