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

with open(f"../config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    model_dict = config['models']

dataset_lst = ['aws', 'azure', 'gcp']

def load_table_acc_em(dataset):
    file_name = f"{project_root}/results/tables/table_em/table_em_{dataset}.csv"
    df = pd.read_csv(file_name)
    return df

def calculate_few_shot_improvement(df):
    """Calculate absolute improvement of few-shot vs zero-shot for each model"""
    df_result = df.copy()
    
    # Calculate improvement for each model
    for model_abbr in model_dict.keys():
        zero_shot_col = f"{model_abbr}-0"
        few_shot_col = f"{model_abbr}-1"
        
        if zero_shot_col in df.columns and few_shot_col in df.columns:
            improvement_col = f"{model_abbr}-improvement"
            df_result[improvement_col] = df[few_shot_col] - df[zero_shot_col]
    
    return df_result

def save_improvement_table(df, dataset):
    """Save the table with improvement calculations"""
    output_file = f"{project_root}/results/tables/table_em/table_em_{dataset}_with_improvement.csv"
    df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Saved improvement table to: {output_file}")

def calculate_improvement_summary(df, dataset):
    """Calculate max, min, and average of all model improvements for a dataset"""
    improvement_cols = [col for col in df.columns if col.endswith('-improvement')]
    
    if not improvement_cols:
        print(f"No improvement columns found for {dataset}")
        return None
    
    # Exclude the average_em row from calculations
    df_filtered = df[df['extract_fields'] != 'average_em']
    
    # Get all improvement values across all models and fields (excluding average_em)
    all_improvements = []
    for col in improvement_cols:
        all_improvements.extend(df_filtered[col].dropna().tolist())
    
    if not all_improvements:
        print(f"No improvement data found for {dataset}")
        return None

    print(f"Expected measurements: {len(improvement_cols)} models × {len(df_filtered)} fields = {len(improvement_cols) * len(df_filtered)}")
    print(f"Actual measurements: {len(all_improvements)}")
    print(f"All improvements: {all_improvements}")
    
    summary = {
        'dataset': dataset,
        'max_improvement': max(all_improvements),
        'min_improvement': min(all_improvements),
        'avg_improvement': np.mean(all_improvements),
        'num_models': len(improvement_cols),
        'total_measurements': len(all_improvements),
        'improve_measurements': len([improvement for improvement in all_improvements if improvement > 0])
    }
    
    return summary

def print_improvement_summary(summary):
    """Print formatted improvement summary"""
    if summary is None:
        return
    
    print(f"\nImprovement Summary for {summary['dataset'].upper()}:")
    print(f"  Number of models: {summary['num_models']}")
    print(f"  Total measurements: {summary['total_measurements']}")
    print(f"  Improve measurements: {summary['improve_measurements']}")
    print(f"  Improve measurements percentage: {summary['improve_measurements'] / summary['total_measurements'] * 100:.2f}%")
    print(f"  Maximum improvement: {summary['max_improvement']:.2f}%")
    print(f"  Minimum improvement: {summary['min_improvement']:.2f}%")
    print(f"  Average improvement: {summary['avg_improvement']:.2f}%")
    print("-" * 50)

# Process all datasets
all_summaries = []

for dataset in dataset_lst:
    print(f"\n{'='*50}")
    print(f"Processing {dataset.upper()} dataset")
    print(f"{'='*50}")
    
    df = load_table_acc_em(dataset)
    df_improved = calculate_few_shot_improvement(df)
    save_improvement_table(df_improved, dataset)
    
    # Calculate and print improvement summary
    summary = calculate_improvement_summary(df_improved, dataset)
    if summary:
        print_improvement_summary(summary)
        all_summaries.append(summary)

# Print overall summary across all datasets
if all_summaries:
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*60}")
    
    all_improvements = []
    for summary in all_summaries:
        all_improvements.extend([
            summary['max_improvement'],
            summary['min_improvement'], 
            summary['avg_improvement']
        ])
    
    print(f"Overall maximum improvement: {max(all_improvements):.2f}%")
    print(f"Overall minimum improvement: {min(all_improvements):.2f}%")
    print(f"Overall average improvement: {np.mean(all_improvements):.2f}%")
    
