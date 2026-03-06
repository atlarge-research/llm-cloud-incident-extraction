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

eval_round_mark = 'eval'
dataset_lst = ['aws', 'azure', 'gcp']
prompt_lst = ['0', '1']
eval_method = 'tk'
plot_fields = ['user_symptom_category']

# select modes
# mode = 'COMBINED' # OR 'SINGLE'
mode = 'COMBINED'

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


plot_round_mark = 'table_tk'

# Create token-level accuracy tables for each dataset
if mode == 'SINGLE':
    print(f"\n{'='*60}")
    print(f"PRODUCING SEPARATE DATASET TABLES (MODE: {mode})")
    print(f"{'='*60}")
    
    for dataset in dataset_lst:
        print(f"\n{'='*60}")
        print(f"CREATING TOKEN-LEVEL ACCURACY TABLE FOR DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        # Define the extraction fields based on the reference notebook
        # Note: If a dataset doesn't contain certain fields, they will be filled with NaN
        extract_fields = plot_fields
        
        # Create comprehensive table with requested column structure
        print(f"\nComprehensive Token-Level Table - {dataset.upper()}")
        print("-" * 120)
        
        # Create comprehensive DataFrame with the requested structure
        comprehensive_data = []
        
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ['0', '1']:
                df_data = all_eval_data[dataset][model_abbr][prompt_type]
                
                if df_data is not None and not df_data.empty:
                    # Get all field values
                    row_data = {
                        'dataset': dataset,
                        'model': model_abbr,
                        'prompt_type': int(prompt_type)
                    }
                    
                    # Add field values for each metric (precision, recall, F1)
                    for field in extract_fields:
                        if field in df_data.columns:
                            try:
                                # Get precision (row 0), recall (row 1), and F1 (row 2)
                                precision_val = df_data.loc[0, field] if 0 < len(df_data) else np.nan
                                recall_val = df_data.loc[1, field] if 1 < len(df_data) else np.nan
                                f1_val = df_data.loc[2, field] if 2 < len(df_data) else np.nan
                                
                                # Store all three metrics
                                row_data[f'{field}_precision'] = precision_val
                                row_data[f'{field}_recall'] = recall_val
                                row_data[f'{field}_f1'] = f1_val
                            except (KeyError, IndexError):
                                row_data[f'{field}_precision'] = np.nan
                                row_data[f'{field}_recall'] = np.nan
                                row_data[f'{field}_f1'] = np.nan
                        else:
                            row_data[f'{field}_precision'] = np.nan
                            row_data[f'{field}_recall'] = np.nan
                            row_data[f'{field}_f1'] = np.nan
                    
                    comprehensive_data.append(row_data)
        
        df_comprehensive = pd.DataFrame(comprehensive_data)
        
        # Display comprehensive table
        print(df_comprehensive)
        
        # Save comprehensive table to CSV
        output_dir = f'{project_root}/results/tables/{plot_round_mark}'
        os.makedirs(output_dir, exist_ok=True)
        comprehensive_csv_path = f'{output_dir}/{plot_round_mark}_{mode.lower()}_{eval_method}_{dataset}.csv'
        df_comprehensive.to_csv(comprehensive_csv_path, index=False)
        print(f"\nSaved comprehensive token-level table to: {comprehensive_csv_path}")
        
        # Generate LaTeX format for comprehensive table
        print("\nComprehensive Table LaTeX Format:")
        print("-" * 60)
        comprehensive_latex_table = df_comprehensive.to_latex(float_format='{:.2f}'.format, index=False)
        # print(comprehensive_latex_table)
        
        # Save comprehensive LaTeX table
        comprehensive_latex_path = f'{output_dir}/{plot_round_mark}_{mode.lower()}_{eval_method}_{dataset}.tex'
        with open(comprehensive_latex_path, 'w') as f:
            f.write(comprehensive_latex_table)
        print(f"Saved comprehensive LaTeX table to: {comprehensive_latex_path}")
        


elif mode == 'COMBINED':
    print(f"\n{'='*60}")
    print(f"PRODUCING COMBINED MULTI-DATASET TABLE (MODE: {mode})")
    print(f"{'='*60}")
    
    # Create combined multi-dataset table
    print(f"\n{'='*80}")
    print(f"CREATING COMBINED MULTI-DATASET TOKEN-LEVEL ACCURACY TABLE")
    print(f"{'='*80}")

    # Define the extraction fields - same as individual tables
    # Note: If a dataset doesn't contain certain fields, they will be filled with NaN
    extract_fields = plot_fields

    # Create comprehensive combined table with requested column structure
    print(f"\nComprehensive Combined Token-Level Table - All Datasets")
    print("-" * 160)
    
    # Create comprehensive DataFrame with the requested structure
    comprehensive_combined_data = []
    
    for dataset in dataset_lst:
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ['0', '1']:
                df_data = all_eval_data[dataset][model_abbr][prompt_type]
                
                if df_data is not None and not df_data.empty:
                    # Get all field values
                    row_data = {
                        'dataset': dataset,
                        'model': model_abbr,
                        'prompt_type': int(prompt_type)
                    }
                    
                    # Add field values for each metric (precision, recall, F1)
                    for field in extract_fields:
                        if field in df_data.columns:
                            try:
                                # Get precision (row 0), recall (row 1), and F1 (row 2)
                                precision_val = df_data.loc[0, field] if 0 < len(df_data) else np.nan
                                precision_val = round(precision_val/100, 4)
                                recall_val = df_data.loc[1, field] if 1 < len(df_data) else np.nan
                                recall_val = round(recall_val/100, 4)
                                f1_val = df_data.loc[2, field] if 2 < len(df_data) else np.nan
                                f1_val = round(f1_val/100, 4)
                                
                                # Store all three metrics
                                row_data[f'{field}_precision'] = precision_val
                                row_data[f'{field}_recall'] = recall_val
                                row_data[f'{field}_f1'] = f1_val
                            except (KeyError, IndexError):
                                row_data[f'{field}_precision'] = np.nan
                                row_data[f'{field}_recall'] = np.nan
                                row_data[f'{field}_f1'] = np.nan
                        else:
                            row_data[f'{field}_precision'] = np.nan
                            row_data[f'{field}_recall'] = np.nan
                            row_data[f'{field}_f1'] = np.nan
                    
                    comprehensive_combined_data.append(row_data)
    
    df_comprehensive_combined = pd.DataFrame(comprehensive_combined_data)
    
    # Display comprehensive combined table
    print(df_comprehensive_combined)
    
    # Save comprehensive combined table to CSV
    output_dir = f'{project_root}/results/tables/{plot_round_mark}'
    os.makedirs(output_dir, exist_ok=True)
    comprehensive_combined_csv_path = f'{output_dir}/{plot_round_mark}_{mode.lower()}.csv'
    df_comprehensive_combined.to_csv(comprehensive_combined_csv_path, index=False)
    print(f"\nSaved comprehensive combined token-level table to: {comprehensive_combined_csv_path}")
    
    # Generate LaTeX format for comprehensive combined table
    print("\nComprehensive Combined Table LaTeX Format:")
    print("-" * 80)
    comprehensive_combined_latex_table = df_comprehensive_combined.to_latex(float_format='{:.4f}'.format, index=False)
    # print(comprehensive_combined_latex_table)
    
    # Save comprehensive combined LaTeX table
    comprehensive_combined_latex_path = f'{output_dir}/{plot_round_mark}_{mode.lower()}.tex'
    with open(comprehensive_combined_latex_path, 'w') as f:
        f.write(comprehensive_combined_latex_table)
    print(f"Saved comprehensive combined LaTeX table to: {comprehensive_combined_latex_path}")



else:
    print(f"ERROR: Invalid mode '{mode}'. Please use 'SINGLE' or 'COMBINED'")
    exit(1)

print("\n" + "="*80)
print("ALL TOKEN-LEVEL ACCURACY TABLES CREATED SUCCESSFULLY!")
print("="*80)
