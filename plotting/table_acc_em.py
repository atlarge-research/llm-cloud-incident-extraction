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

from utils.plotting import load_evaluation_csv


with open(f"../config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    model_dict = config['models']

## for testing
# model_dict = {'gpt-3.5': 'gpt-3.5-turbo',
#  'gpt-4o': 'gpt-4o',
#  'claude-3-5': 'claude-3-5-haiku-20241022',
#  'claude-4': 'claude-sonnet-4-20250514',
#  'gemini-2.0': 'gemini-2.0-flash',
#  'gemini-2.5': 'gemini-2.5-pro'}

eval_round_mark = 'eval'
dataset_lst = ['azure']
prompt_lst = ['0', '1']
eval_method = 'em'

# select modes
# mode = 'COMBINED' # OR 'SINGLE'
mode = 'SINGLE'

# load evaluation data of all datasets, models, and prompt types
print("Loading evaluation data for all datasets, models, and prompt types...")

# Dictionary to store all evaluation data
all_eval_data = {}

for dataset in dataset_lst:
    print(f"\nProcessing dataset: {dataset}")
    all_eval_data[dataset] = {}
    
    # Load evaluation data for all models and prompt types
    for model_abbr, model_name in model_dict.items():
        all_eval_data[dataset][model_abbr] = {}
        
        # Load zero-shot (prompt_type=0)
        try:
            df_zero = load_evaluation_csv(project_root, eval_round_mark, dataset, model_abbr, model_name, prompt_type=0, eval_method=eval_method)
            all_eval_data[dataset][model_abbr]['0'] = df_zero
            print(f"  ✓ Loaded {model_abbr} zero-shot data")
        except Exception as e:
            print(f"  ✗ Failed to load {model_abbr} zero-shot data: {e}")
            all_eval_data[dataset][model_abbr]['0'] = None
        
        # Load few-shot (prompt_type=1)
        try:
            df_few = load_evaluation_csv(project_root, eval_round_mark, dataset, model_abbr, model_name, prompt_type=1, eval_method=eval_method)
            all_eval_data[dataset][model_abbr]['1'] = df_few
            print(f"  ✓ Loaded {model_abbr} few-shot data")
        except Exception as e:
            print(f"  ✗ Failed to load {model_abbr} few-shot data: {e}")
            all_eval_data[dataset][model_abbr]['1'] = None

print("\n" + "="*80)
print("EVALUATION DATA LOADING COMPLETED")
print("="*80)

plot_round_mark = 'table_em'

# Create EM accuracy tables for each dataset
if mode == 'SINGLE':
    print(f"\n{'='*60}")
    print(f"PRODUCING SEPARATE DATASET TABLES (MODE: {mode})")
    print(f"{'='*60}")
    
    for dataset in dataset_lst:
        print(f"\n{'='*60}")
        print(f"CREATING EM ACCURACY TABLE FOR DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        # Create DataFrame for EM accuracy
        cols = []
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ['0', '1']:
                cols.append(f'{model_abbr}-{prompt_type}')
        df_em = pd.DataFrame(columns=['extract_fields'] + cols)
        
        # Define the extraction fields based on the reference notebook
        # Note: If a dataset doesn't contain certain fields, they will be filled with NaN
        df_em['extract_fields'] = ['service_name', 'location', 'start_time', 'end_time', 'timezone', 
                                   'service_category', 'root_cause_category']
        df_em.set_index('extract_fields', inplace=True)
        
        # Fill data for each model and prompt type
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ['0', '1']:
                df_data = all_eval_data[dataset][model_abbr][prompt_type]
                
                if df_data is not None and not df_data.empty:
                    # Get the EM accuracy values (first row, index 0)
                    for field in df_em.index:
                        if field in df_data.columns:
                            try:
                                # Get the EM accuracy value from the first row
                                em_value = df_data.loc[0, field]
                                df_em[f'{model_abbr}-{prompt_type}'].loc[field] = em_value
                            except (KeyError, IndexError):
                                df_em[f'{model_abbr}-{prompt_type}'].loc[field] = np.nan
                        else:
                            df_em[f'{model_abbr}-{prompt_type}'].loc[field] = np.nan
                else:
                    # If no data available, fill with NaN
                    df_em[f'{model_abbr}-{prompt_type}'] = np.nan
        
        # Add average EM accuracy at the last row
        df_em.loc['average_em'] = df_em.mean(axis=0)
        
        # Convert to float and round to 2 decimal places
        df_em = df_em.astype(float).round(2)
        
        # Display the table
        print(f'\nTable: Exact Match Accuracy - {dataset.upper()}')
        print("-" * 80)
        print(df_em)
        
        # Save to CSV
        output_dir = f'{project_root}/results/tables/{plot_round_mark}'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f'{output_dir}/{plot_round_mark}_{dataset}.csv'
        df_em.to_csv(csv_path)
        print(f"\nSaved EM accuracy table to: {csv_path}")
        
        # Generate LaTeX format
        print("\nLaTeX Format:")
        print("-" * 40)
        latex_table = df_em.to_latex(float_format='{:.2f}'.format)
        # Save LaTeX table
        latex_path = f'{output_dir}/{plot_round_mark}_{dataset}.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved LaTeX table to: {latex_path}")
        
        # Display summary statistics
        print(f"\nSummary for {dataset.upper()} (average accuracy):")
        print("-" * 40)
        summary_stats = ""
        for model_abbr, model_name in model_dict.items():
            zero_shot_avg = df_em.loc['average_em', f'{model_abbr}-0']
            few_shot_avg = df_em.loc['average_em', f'{model_abbr}-1']
            improvement = few_shot_avg - zero_shot_avg
            improvement_pct = (improvement / zero_shot_avg * 100) if zero_shot_avg != 0 else 0
            
            # print(f"{model_abbr}:")
            print(f"  {model_abbr}: {zero_shot_avg:.2f}%(zero-shot) → {few_shot_avg:.2f}%(few-shot) ({improvement:+.2f}%, {improvement_pct:+.1f}%)")
            summary_stats += f"{model_abbr}: {zero_shot_avg:.2f}%(zero-shot) → {few_shot_avg:.2f}%(few-shot) ({improvement:+.2f}%, {improvement_pct:+.1f}%)\n"
        # save summary statistics to csv
        summary_stats_path = f'{output_dir}/{plot_round_mark}_{dataset}_summary_stats.txt'
        with open(summary_stats_path, 'w') as f:
            f.write(summary_stats)
        print(f"Saved summary statistics to: {summary_stats_path}")

elif mode == 'COMBINED':
    print(f"\n{'='*60}")
    print(f"PRODUCING COMBINED MULTI-DATASET TABLE (MODE: {mode})")
    print(f"{'='*60}")
    
    # Create combined multi-dataset table
    print(f"\n{'='*80}")
    print(f"CREATING COMBINED MULTI-DATASET EM ACCURACY TABLE")
    print(f"{'='*80}")

    # Create combined DataFrame with all datasets
    combined_columns = []
    for dataset in dataset_lst:
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ['0', '1']:
                combined_columns.append(f'{dataset}_{model_abbr}-{prompt_type}')

    df_combined = pd.DataFrame(columns=['extract_fields'] + combined_columns)

    # Define the extraction fields - same as individual tables
    # Note: If a dataset doesn't contain certain fields, they will be filled with NaN
    df_combined['extract_fields'] = ['service_name', 'location', 'start_time', 'end_time', 'timezone', 
                                   'service_category', 'root_cause_category']
    df_combined.set_index('extract_fields', inplace=True)

    # Fill data for combined table
    for dataset in dataset_lst:
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ['0', '1']:
                df_data = all_eval_data[dataset][model_abbr][prompt_type]
                col_name = f'{dataset}_{model_abbr}-{prompt_type}'
                
                if df_data is not None and not df_data.empty:
                    for field in df_combined.index:
                        if field in df_data.columns:
                            try:
                                em_value = df_data.loc[0, field]
                                df_combined[col_name].loc[field] = em_value
                            except (KeyError, IndexError):
                                df_combined[col_name].loc[field] = np.nan
                        else:
                            df_combined[col_name].loc[field] = np.nan
                else:
                    df_combined[col_name] = np.nan

    # Add average EM accuracy for each dataset-model-prompt combination
    df_combined.loc['average_em'] = df_combined.mean(axis=0)

    # Convert to float and round to 2 decimal places
    df_combined = df_combined.astype(float).round(2)

    # Display the combined table
    print(f'\nCombined Table: Exact Match Accuracy - All Datasets')
    print("-" * 120)
    print(df_combined)

    # Save combined table to CSV
    output_dir = f'{project_root}/results/tables/table_em'
    os.makedirs(output_dir, exist_ok=True)
    combined_csv_path = f'{output_dir}/{plot_round_mark}_{mode.lower()}.csv'
    df_combined.to_csv(combined_csv_path)
    print(f"\nSaved combined EM accuracy table to: {combined_csv_path}")

    # Generate LaTeX format for combined table
    print("\nCombined Table LaTeX Format:")
    print("-" * 60)
    combined_latex_table = df_combined.to_latex(float_format='{:.2f}'.format)
    # print(combined_latex_table)

    # Save combined LaTeX table
    combined_latex_path = f'{output_dir}/{plot_round_mark}_combined.tex'
    with open(combined_latex_path, 'w') as f:
        f.write(combined_latex_table)
    print(f"Saved combined LaTeX table to: {combined_latex_path}")

    # Display cross-dataset summary statistics
    print(f"\nCross-Dataset Summary:")
    print("-" * 60)
    for model_abbr, model_name in model_dict.items():
        print(f"\n{model_abbr}:")
        
        # Calculate averages across all datasets for each prompt type
        zero_shot_cols = [f'{dataset}_{model_abbr}-0' for dataset in dataset_lst]
        few_shot_cols = [f'{dataset}_{model_abbr}-1' for dataset in dataset_lst]
        
        zero_shot_avg = df_combined.loc['average_em', zero_shot_cols].mean()
        few_shot_avg = df_combined.loc['average_em', few_shot_cols].mean()
        improvement = few_shot_avg - zero_shot_avg
        improvement_pct = (improvement / zero_shot_avg * 100) if zero_shot_avg != 0 else 0
        
        print(f"  CROSS-DATASET: {zero_shot_avg:.2f}%(zero-shot) → {few_shot_avg:.2f}%(few-shot) ({improvement:+.2f}%, {improvement_pct:+.1f}%)")

        
        # # Show individual dataset performance
        # for dataset in dataset_lst:
        #     zero_val = df_combined.loc['average_em', f'{dataset}_{model_abbr}-0']
        #     few_val = df_combined.loc['average_em', f'{dataset}_{model_abbr}-1']
        #     dataset_improvement = few_val - zero_val
        #     dataset_improvement_pct = (dataset_improvement / zero_val * 100) if zero_val != 0 else 0
            
        #     print(f"  {dataset.upper()}: {zero_val:.2f}%(zero-shot) → {few_val:.2f}%(few-shot) ({dataset_improvement:+.2f}%, {dataset_improvement_pct:+.1f}%)")

else:
    print(f"ERROR: Invalid mode '{mode}'. Please use 'SINGLE' or 'COMBINED'")
    exit(1)

print("\n" + "="*80)
print("ALL EM ACCURACY TABLES CREATED SUCCESSFULLY!")
print("="*80)
