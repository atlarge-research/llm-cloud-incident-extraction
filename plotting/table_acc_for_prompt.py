from re import T
import numpy as np
import pandas as pd
import yaml

import warnings
warnings.filterwarnings('ignore')

import os
import sys
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.insert(0, project_root)

from utils.plotting import load_evaluation_csv_for_prompt


model_dict = {'gpt-3.5': 'gpt-3.5-turbo'}


eval_round_mark = 'pmpt'
dataset_lst = ['aws']
prompt_strategy_type_lst = ["full-zs", "full-fs", "basic-zs", "basic-fs", "cot-zs", "categ-zs"]
eval_method_lst = ['em', 'tk', 'bs']

# load evaluation data of prompt strategy types
print("Loading evaluation data for prompt strategy types, eval methods...")

# Dictionary to store all evaluation data for each prompt strategy type and eval method
all_eval_data = {}
for prompt_strategy_type in prompt_strategy_type_lst:
    all_eval_data[prompt_strategy_type] = {}
    for eval_method in eval_method_lst:
        all_eval_data[prompt_strategy_type][eval_method] = {}
        try:
            df = load_evaluation_csv_for_prompt(project_root, eval_round_mark, dataset='aws', model_abbr='gpt-3.5', model_name='gpt-3.5-turbo', prompt_strategy_type=prompt_strategy_type, eval_method=eval_method)
            all_eval_data[prompt_strategy_type][eval_method] = df
            print(f"  ✓ Loaded gpt-3.5 {prompt_strategy_type} {eval_method} data")
        except Exception as e:
            print(f"  ✗ Failed to load gpt-3.5 {prompt_strategy_type} {eval_method} data: {e}")
            all_eval_data[prompt_strategy_type][eval_method] = None

print("\n" + "="*80)
print("EVALUATION DATA LOADING COMPLETED")
print("="*80)

plot_round_mark = 'table_acc_for_prompt'

em_fields = ['service_name', 'location', 'start_time', 'end_time', 'timezone', 'service_category']
tk_fields = ['user_symptom_category']
bs_fields = ['user_symptom']

# Create combined table where each field shows only its corresponding eval method
# Columns are prompt strategy types
cols = prompt_strategy_type_lst
all_fields = em_fields + tk_fields + bs_fields
df_combined = pd.DataFrame(columns=['extract_fields', 'eval_method'] + cols)
df_combined['extract_fields'] = all_fields

# Map each field to its evaluation method
field_to_eval_method = {}
for field in em_fields:
    field_to_eval_method[field] = 'em'
for field in tk_fields:
    field_to_eval_method[field] = 'tk'
for field in bs_fields:
    field_to_eval_method[field] = 'bs'

df_combined['eval_method'] = df_combined['extract_fields'].map(field_to_eval_method)
df_combined.set_index('extract_fields', inplace=True)

# Fill in the values: for each field, use only its corresponding eval method
for field in all_fields:
    eval_method = field_to_eval_method[field]
    for prompt_strategy_type in prompt_strategy_type_lst:
        df_data = all_eval_data[prompt_strategy_type][eval_method]
        if df_data is not None and not df_data.empty:
            if field in df_data.columns:
                try:
                    value = df_data.loc[0, field]
                    df_combined.loc[field, prompt_strategy_type] = value
                except (KeyError, IndexError):
                    df_combined.loc[field, prompt_strategy_type] = np.nan
            else:
                df_combined.loc[field, prompt_strategy_type] = np.nan

# Calculate average accuracy for each prompt strategy column
average_row = {}
for prompt_strategy_type in prompt_strategy_type_lst:
    # Get numeric values (excluding NaN) for averaging
    numeric_values = df_combined[prompt_strategy_type].dropna()
    if len(numeric_values) > 0:
        # For BS values, they're already in 0-1 range, so we can average directly
        # For EM and TK, they might be percentages or decimals - we'll average as-is
        avg_value = numeric_values.mean()
        average_row[prompt_strategy_type] = round(avg_value, 2)
    else:
        average_row[prompt_strategy_type] = np.nan

# Add average row to dataframe
average_row['eval_method'] = 'average'
df_combined.loc['average_acc'] = average_row

# Format BS values as percentages with 2 decimal places
for field in bs_fields:
    for prompt_strategy_type in prompt_strategy_type_lst:
        value = df_combined.loc[field, prompt_strategy_type]
        if pd.notna(value):
            df_combined.loc[field, prompt_strategy_type] = f"{value * 100:.2f}"


print(df_combined)

# Save to CSV and LaTeX
output_dir = f'{project_root}/results/prompts/tables/{plot_round_mark}'
os.makedirs(output_dir, exist_ok=True)
csv_path = f'{output_dir}/{plot_round_mark}.csv'
latex_path = f'{output_dir}/{plot_round_mark}.tex'
df_combined.to_csv(csv_path, index=True)
df_combined.to_latex(latex_path, index=True)
print(f"Saved prompt accuracy table to: {csv_path}")
print(f"Saved prompt accuracy table to: {latex_path}")