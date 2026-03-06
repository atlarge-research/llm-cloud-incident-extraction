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

from utils.plotting import load_evaluation_csv, load_model_perf_csv

with open(f"{project_root}/config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    model_dict = config['models']

eval_round_mark = 'eval'
dataset_lst = ['aws', 'azure', 'gcp']
prompt_lst = ['0', '1']

# Configuration for which accuracy columns to include in CSV output
# Options: 'acc_em', 'acc_tk', 'acc_bs', 'avg_acc'
# Set to True to include the column, False to exclude
ACCURACY_COLUMNS = {
    'acc_em': False,    # EM accuracy (converted to decimal)
    'acc_tk': False,    # Token F1 accuracy
    'acc_bs': False,    # BertScore F1 accuracy
    'avg_acc': True    # Average of all three accuracy metrics
}


# Load evaluation data for EM accuracy (needed for performance analysis)
print("Loading evaluation data for EM accuracy...")
eval_method = 'em'
all_eval_data_em = {}
for dataset in dataset_lst:
    print(f"\nProcessing dataset: {dataset}")
    all_eval_data_em[dataset] = {}
    for model_abbr, model_name in model_dict.items():
        all_eval_data_em[dataset][model_abbr] = {}
        # Load zero-shot (prompt_type=0)
        try:
            df_zero = load_evaluation_csv(project_root, eval_round_mark, dataset, model_abbr, model_name, prompt_type=0, eval_method=eval_method)
            all_eval_data_em[dataset][model_abbr]['0'] = df_zero
            print(f"  ✓ Loaded {model_abbr} zero-shot em accuracy data")
        except Exception as e:
            print(f"  ✗ Failed to load {model_abbr} zero-shot em accuracy data: {e}")
            all_eval_data_em[dataset][model_abbr]['0'] = None
        
        # Load few-shot (prompt_type=1)
        try:
            df_few = load_evaluation_csv(project_root, eval_round_mark, dataset, model_abbr, model_name, prompt_type=1, eval_method=eval_method)
            all_eval_data_em[dataset][model_abbr]['1'] = df_few
            print(f"  ✓ Loaded {model_abbr} few-shot em accuracy data")
        except Exception as e:
            print(f"  ✗ Failed to load {model_abbr} few-shot em accuracy data: {e}")
            all_eval_data_em[dataset][model_abbr]['1'] = None

print("\n" + "="*80)
print("EM ACCURACY DATA LOADING COMPLETED")
print("="*80)

# Load evaluation data for tk F1 score (needed for performance analysis)
print("Loading evaluation data for Token F1 score...")
eval_method = 'tk'
all_eval_data_tk = pd.read_csv(f"{project_root}/results/tables/table_tk/table_tk_combined.csv")

# Load evaluation data for bs score (needed for performance analysis)
print("Loading evaluation data for BertScore...")
eval_method = 'bs'
all_eval_data_bs = pd.read_csv(f"{project_root}/results/tables/table_bs/table_bs_combined.csv")

# Calculate acc_tk and acc_bs columns
print("Calculating acc_tk and acc_bs columns...")

# For TK table: acc_tk = user_symptom_category_f1
all_eval_data_tk['acc_tk'] = all_eval_data_tk['user_symptom_category_f1']

# For BS table: acc_bs = average of user_symptom_f1 and root_cause_f1 (where available)
def calculate_bs_accuracy(row):
    user_f1 = row['user_symptom_f1']
    root_f1 = row['root_cause_f1']
    
    # If root_cause_f1 is NaN or empty, just use user_symptom_f1
    if pd.isna(root_f1) or root_f1 == '':
        return user_f1
    else:
        # Average of both F1 scores
        return (user_f1 + root_f1) / 2

all_eval_data_bs['acc_bs'] = all_eval_data_bs.apply(calculate_bs_accuracy, axis=1)

print("✓ Added acc_tk and acc_bs columns to evaluation data")




            
print("\n" + "="*80)





# Load performance data (results/extractions)
ext_round_mark = 'ext'
print("\nLoading performance data...")
all_perf_data = {}
for dataset in dataset_lst:
    print(f"\nProcessing performance data for dataset: {dataset}")
    all_perf_data[dataset] = {}
    for model_abbr, model_name in model_dict.items():
        all_perf_data[dataset][model_abbr] = {}
        
        # Load zero-shot performance data
        try:
            df_zero = load_model_perf_csv(project_root, ext_round_mark, dataset, model_abbr, model_name, prompt_type=0)
            all_perf_data[dataset][model_abbr]['0'] = df_zero
            print(f"  ✓ Loaded {model_abbr} zero-shot performance data")
        except Exception as e:
            print(f"  ✗ Failed to load {model_abbr} zero-shot performance data: {e}")
            all_perf_data[dataset][model_abbr]['0'] = None
        
        # Load few-shot performance data
        try:
            df_few = load_model_perf_csv(project_root, ext_round_mark, dataset, model_abbr, model_name, prompt_type=1)
            all_perf_data[dataset][model_abbr]['1'] = df_few
            print(f"  ✓ Loaded {model_abbr} few-shot performance data")
        except Exception as e:
            print(f"  ✗ Failed to load {model_abbr} few-shot performance data: {e}")
            all_perf_data[dataset][model_abbr]['1'] = None

print("\n" + "="*80)
print("PERFORMANCE DATA LOADING COMPLETED")
print("="*80)

def calculate_model_averages(dfs_dict, models):
    """Calculate average latency, tokens, and cost for all models"""
    averages = {}
    for model in models:
        if dfs_dict[model] is not None and not dfs_dict[model].empty:
            df = dfs_dict[model]
            averages[model] = {
                'latency': df['latency(s)'].mean(),
                'input_tokens': df['token_input'].mean(),
                'output_tokens': df['token_output'].mean(),
                'tokens': df['token_input'].mean() + df['token_output'].mean(),
                'input_cost': df['cost_input(USD)'].mean(),
                'output_cost': df['cost_output(USD)'].mean()
            }
        else:
            averages[model] = {
                'latency': np.nan,
                'input_tokens': np.nan,
                'output_tokens': np.nan,
                'tokens': np.nan,
                'input_cost': np.nan,
                'output_cost': np.nan
            }
    return averages

# Create performance tables for each dataset
table_round_mark = 'table_perf_cost'

print(f"\n{'='*60}")
print(f"PRODUCING COMBINED PERFORMANCE TABLE (MODE: {table_round_mark})")
print(f"{'='*60}")

# Create combined performance DataFrame
combined_perf_data = []

for dataset in dataset_lst:
    # Get EM accuracy averages for this dataset
    df_em = pd.DataFrame(columns=['extract_fields', 'gpt-3.5-0', 'gpt-3.5-1', 'gpt-4o-0', 'gpt-4o-1',
                                    'claude-3-5-0', 'claude-3-5-1', 'claude-4-0', 'claude-4-1',
                                    'gemini-2.0-0', 'gemini-2.0-1', 'gemini-2.5-0', 'gemini-2.5-1'])
    
    df_em['extract_fields'] = ['service_name', 'location', 'start_time', 'end_time', 'timezone', 
                                'service_category', 'user_symptom_category', 'user_symptom', 'root_cause', 'root_cause_category']
    df_em.set_index('extract_fields', inplace=True)
    
    # Fill EM accuracy data
    for model_abbr, model_name in model_dict.items():
        for prompt_type in ['0', '1']:
            df_data = all_eval_data_em[dataset][model_abbr][prompt_type]
            
            if df_data is not None and not df_data.empty:
                for field in df_em.index:
                    if field in df_data.columns:
                        try:
                            em_value = df_data.loc[0, field]
                            df_em[f'{model_abbr}-{prompt_type}'].loc[field] = em_value
                        except (KeyError, IndexError):
                            df_em[f'{model_abbr}-{prompt_type}'].loc[field] = np.nan
                    else:
                        df_em[f'{model_abbr}-{prompt_type}'].loc[field] = np.nan
            else:
                df_em[f'{model_abbr}-{prompt_type}'] = np.nan
    
    # Add average EM accuracy
    df_em.loc['average_em'] = df_em.mean(axis=0)
    df_em = df_em.astype(float).round(2)
    
    # Calculate performance averages
    models = list(model_dict.keys())
    avg_zero = calculate_model_averages({model_abbr: all_perf_data[dataset][model_abbr]['0'] for model_abbr in models}, models)
    avg_few = calculate_model_averages({model_abbr: all_perf_data[dataset][model_abbr]['1'] for model_abbr in models}, models)
    
    # Get acc_tk and acc_bs values for this dataset
    tk_data = all_eval_data_tk[all_eval_data_tk['dataset'] == dataset]
    bs_data = all_eval_data_bs[all_eval_data_bs['dataset'] == dataset]
    
    # Add to combined data
    for model_abbr in models:
        # Get acc_tk and acc_bs values for this model and prompt type
        tk_zero = tk_data[(tk_data['model'] == model_abbr) & (tk_data['prompt_type'] == 0)]['acc_tk'].iloc[0] if not tk_data[(tk_data['model'] == model_abbr) & (tk_data['prompt_type'] == 0)].empty else np.nan
        tk_few = tk_data[(tk_data['model'] == model_abbr) & (tk_data['prompt_type'] == 1)]['acc_tk'].iloc[0] if not tk_data[(tk_data['model'] == model_abbr) & (tk_data['prompt_type'] == 1)].empty else np.nan
        bs_zero = bs_data[(bs_data['model'] == model_abbr) & (bs_data['prompt_type'] == 0)]['acc_bs'].iloc[0] if not bs_data[(bs_data['model'] == model_abbr) & (bs_data['prompt_type'] == 0)].empty else np.nan
        bs_few = bs_data[(bs_data['model'] == model_abbr) & (bs_data['prompt_type'] == 1)]['acc_bs'].iloc[0] if not bs_data[(bs_data['model'] == model_abbr) & (bs_data['prompt_type'] == 1)].empty else np.nan
        
        # Calculate acc_em as percentage (divide by 100)
        acc_em_zero = df_em.loc['average_em', f'{model_abbr}-0'] / 100
        acc_em_few = df_em.loc['average_em', f'{model_abbr}-1'] / 100
        
        # Calculate avg_acc as average of acc_em, acc_tk, and acc_bs
        avg_acc_zero = (acc_em_zero + tk_zero + bs_zero) / 3
        avg_acc_few = (acc_em_few + tk_few + bs_few) / 3
        
        # Zero-shot data
        combined_perf_data.append({
            'dataset': dataset,
            'model': model_abbr,
            'prompt_type': 0,
            'acc_em': acc_em_zero,
            'acc_tk': tk_zero,
            'acc_bs': bs_zero,
            'avg_acc': avg_acc_zero,
            'avg_latency': avg_zero[model_abbr]['latency'],
            'avg_input_tokens': avg_zero[model_abbr]['input_tokens'],
            'avg_output_tokens': avg_zero[model_abbr]['output_tokens'],
            'avg_total_tokens': avg_zero[model_abbr]['tokens'],
            'avg_input_cost_10m4': avg_zero[model_abbr]['input_cost'] * 10000,
            'avg_output_cost_10m4': avg_zero[model_abbr]['output_cost'] * 10000,
            'avg_total_cost_10m4': (avg_zero[model_abbr]['input_cost'] + avg_zero[model_abbr]['output_cost']) * 10000
        })
        # Few-shot data
        combined_perf_data.append({
            'dataset': dataset,
            'model': model_abbr,
            'prompt_type': 1,
            'acc_em': acc_em_few,
            'acc_tk': tk_few,
            'acc_bs': bs_few,
            'avg_acc': avg_acc_few,
            'avg_latency': avg_few[model_abbr]['latency'],
            'avg_input_tokens': avg_few[model_abbr]['input_tokens'],
            'avg_output_tokens': avg_few[model_abbr]['output_tokens'],
            'avg_total_tokens': avg_few[model_abbr]['tokens'],
            'avg_input_cost_10m4': avg_few[model_abbr]['input_cost'] * 10000,
            'avg_output_cost_10m4': avg_few[model_abbr]['output_cost'] * 10000,
            'avg_total_cost_10m4': (avg_few[model_abbr]['input_cost'] + avg_few[model_abbr]['output_cost']) * 10000
        })

df_combined_perf = pd.DataFrame(combined_perf_data)

# Display the combined performance table
print(f'\nCombined Performance Table: All Datasets')
print("-" * 220)
print(df_combined_perf[['dataset', 'model', 'prompt_type', 'acc_em', 'acc_tk', 'acc_bs', 'avg_acc', 'avg_latency', 'avg_input_tokens', 'avg_output_tokens', 'avg_total_tokens', 'avg_input_cost_10m4', 'avg_output_cost_10m4', 'avg_total_cost_10m4']])

    # Save combined table to CSV
output_dir = f'{project_root}/results/tables/{table_round_mark}'
os.makedirs(output_dir, exist_ok=True)
combined_csv_path = f'{output_dir}/{table_round_mark}.csv'

# Build column list for CSV based on configuration
csv_columns = ['dataset', 'model', 'prompt_type']
if ACCURACY_COLUMNS['acc_em']:
    csv_columns.append('acc_em')
if ACCURACY_COLUMNS['acc_tk']:
    csv_columns.append('acc_tk')
if ACCURACY_COLUMNS['acc_bs']:
    csv_columns.append('acc_bs')
if ACCURACY_COLUMNS['avg_acc']:
    csv_columns.append('avg_acc')

# Add performance and cost columns
csv_columns.extend(['avg_latency', 'avg_input_tokens', 'avg_output_tokens', 'avg_total_tokens', 
                   'avg_input_cost_10m4', 'avg_output_cost_10m4', 'avg_total_cost_10m4'])

# Save only the selected columns
df_combined_perf[csv_columns].to_csv(combined_csv_path, index=False, float_format='{:.2f}'.format)
print(f"\nSaved combined performance table to: {combined_csv_path}")
print(f"Included accuracy columns: {[col for col in ['acc_em', 'acc_tk', 'acc_bs', 'avg_acc'] if ACCURACY_COLUMNS[col]]}")

# Generate LaTeX format for combined table
print("\nCombined Table LaTeX Format:")
print("-" * 60)
combined_latex_table = df_combined_perf.to_latex(float_format='{:.2f}'.format, index=False)
# print(combined_latex_table)

# Save LaTeX table
combined_latex_path = f'{output_dir}/{table_round_mark}.tex'
with open(combined_latex_path, 'w') as f:
    f.write(combined_latex_table)
print(f"Saved combined LaTeX table to: {combined_latex_path}")

# Display cross-dataset summary statistics
print(f"\nCross-Dataset Performance Summary:")
print("-" * 80)
for model_abbr, model_name in model_dict.items():
    print(f"\n{model_name}:")
    
    # Calculate averages across all datasets for each prompt type
    zero_data = df_combined_perf[(df_combined_perf['model'] == model_abbr) & (df_combined_perf['prompt_type'] == 0)]
    few_data = df_combined_perf[(df_combined_perf['model'] == model_abbr) & (df_combined_perf['prompt_type'] == 1)]
    
    if not zero_data.empty and not few_data.empty:
        # Calculate performance metrics
        zero_latency_avg = zero_data['avg_latency'].mean()
        few_latency_avg = few_data['avg_latency'].mean()
        zero_input_cost_avg = zero_data['avg_input_cost_10m4'].mean()
        few_input_cost_avg = few_data['avg_input_cost_10m4'].mean()
        zero_output_cost_avg = zero_data['avg_output_cost_10m4'].mean()
        few_output_cost_avg = few_data['avg_output_cost_10m4'].mean()
        
        # Calculate percentage changes for performance metrics
        latency_change = few_latency_avg - zero_latency_avg
        latency_change_pct = (latency_change / zero_latency_avg * 100) if zero_latency_avg != 0 else 0
        input_cost_change = few_input_cost_avg - zero_input_cost_avg
        input_cost_change_pct = (input_cost_change / zero_input_cost_avg * 100) if zero_input_cost_avg != 0 else 0
        output_cost_change = few_output_cost_avg - zero_output_cost_avg
        output_cost_change_pct = (output_cost_change / zero_output_cost_avg * 100) if zero_output_cost_avg != 0 else 0
        
        total_zero_cost_avg = zero_input_cost_avg + zero_output_cost_avg
        total_few_cost_avg = few_input_cost_avg + few_output_cost_avg
        total_cost_change = total_few_cost_avg - total_zero_cost_avg
        total_cost_change_pct = (total_cost_change / total_zero_cost_avg * 100) if total_zero_cost_avg != 0 else 0
        
        print(f"  CROSS-DATASET AVERAGES:")
        
        # Print accuracy statistics for enabled columns
        if ACCURACY_COLUMNS['acc_em'] and 'acc_em' in zero_data.columns:
            zero_em_avg = zero_data['acc_em'].mean()
            few_em_avg = few_data['acc_em'].mean()
            em_improvement = few_em_avg - zero_em_avg
            em_improvement_pct = (em_improvement / zero_em_avg * 100) if zero_em_avg != 0 else 0
            print(f"    EM Accuracy: {zero_em_avg:.2f} → {few_em_avg:.2f} ({em_improvement:+.2f}, {em_improvement_pct:+.1f}%)")
        
        if ACCURACY_COLUMNS['acc_tk'] and 'acc_tk' in zero_data.columns:
            zero_tk_avg = zero_data['acc_tk'].mean()
            few_tk_avg = few_data['acc_tk'].mean()
            tk_improvement = few_tk_avg - zero_tk_avg
            tk_improvement_pct = (tk_improvement / zero_tk_avg * 100) if zero_tk_avg != 0 else 0
            print(f"    TK Accuracy: {zero_tk_avg:.3f} → {few_tk_avg:.3f} ({tk_improvement:+.3f}, {tk_improvement_pct:+.1f}%)")
        
        if ACCURACY_COLUMNS['acc_bs'] and 'acc_bs' in zero_data.columns:
            zero_bs_avg = zero_data['acc_bs'].mean()
            few_bs_avg = few_data['acc_bs'].mean()
            bs_improvement = few_bs_avg - zero_bs_avg
            bs_improvement_pct = (bs_improvement / zero_bs_avg * 100) if zero_bs_avg != 0 else 0
            print(f"    BS Accuracy: {zero_bs_avg:.3f} → {few_bs_avg:.3f} ({bs_improvement:+.3f}, {bs_improvement_pct:+.1f}%)")
        
        if ACCURACY_COLUMNS['avg_acc'] and 'avg_acc' in zero_data.columns:
            zero_acc_avg = zero_data['avg_acc'].mean()
            few_acc_avg = few_data['avg_acc'].mean()
            acc_improvement = few_acc_avg - zero_acc_avg
            acc_improvement_pct = (acc_improvement / zero_acc_avg * 100) if zero_acc_avg != 0 else 0
            print(f"    Avg Accuracy: {zero_acc_avg:.3f} → {few_acc_avg:.3f} ({acc_improvement:+.3f}, {acc_improvement_pct:+.1f}%)")
        
        # Print performance metrics
        print(f"    Latency: {zero_latency_avg:.2f}s → {few_latency_avg:.2f}s ({latency_change:+.2f}s, {latency_change_pct:+.1f}%)")
        print(f"    Input Cost: ${zero_input_cost_avg:.2f} → ${few_input_cost_avg:.2f} ({input_cost_change:+.2f}, {input_cost_change_pct:+.1f}%)")
        print(f"    Output Cost: ${zero_output_cost_avg:.2f} → ${few_output_cost_avg:.2f} ({output_cost_change:+.2f}, {output_cost_change_pct:+.1f}%)")
        print(f"    Total Cost: ${total_zero_cost_avg:.2f} → ${total_few_cost_avg:.2f} ({total_cost_change:+.2f}, {total_cost_change_pct:+.1f}%)")
        
        # Show individual dataset performance
        for dataset in dataset_lst:
            dataset_zero = zero_data[zero_data['dataset'] == dataset]
            dataset_few = few_data[few_data['dataset'] == dataset]
            
            if not dataset_zero.empty and not dataset_few.empty:
                # Get performance metrics
                zero_latency = dataset_zero['avg_latency'].iloc[0]
                few_latency = dataset_few['avg_latency'].iloc[0]
                zero_input_cost = dataset_zero['avg_input_cost_10m4'].iloc[0]
                few_input_cost = dataset_few['avg_input_cost_10m4'].iloc[0]
                zero_output_cost = dataset_zero['avg_output_cost_10m4'].iloc[0]
                few_output_cost = dataset_few['avg_output_cost_10m4'].iloc[0]
                
                # Calculate percentage changes for performance metrics
                latency_change = few_latency - zero_latency
                latency_change_pct = (latency_change / zero_latency * 100) if zero_latency != 0 else 0
                input_cost_change = few_input_cost - zero_input_cost
                input_cost_change_pct = (input_cost_change / zero_input_cost * 100) if zero_input_cost != 0 else 0
                output_cost_change = few_output_cost - zero_output_cost
                output_cost_change_pct = (output_cost_change / zero_output_cost * 100) if zero_output_cost != 0 else 0
                
                total_zero_cost = zero_input_cost + zero_output_cost
                total_few_cost = few_input_cost + few_output_cost
                total_cost_change = total_few_cost - total_zero_cost
                total_cost_change_pct = (total_cost_change / total_zero_cost * 100) if total_zero_cost != 0 else 0
                
                print(f"  {dataset.upper()}:")
                
                # Print accuracy statistics for enabled columns
                if ACCURACY_COLUMNS['acc_em'] and 'acc_em' in dataset_zero.columns:
                    zero_em = dataset_zero['acc_em'].iloc[0]
                    few_em = dataset_few['acc_em'].iloc[0]
                    em_improvement = few_em - zero_em
                    em_improvement_pct = (em_improvement / zero_em * 100) if zero_em != 0 else 0
                    print(f"    EM Accuracy: {zero_em:.2f} → {few_em:.2f} ({em_improvement:+.2f}, {em_improvement_pct:+.1f}%)")
                
                if ACCURACY_COLUMNS['acc_tk'] and 'acc_tk' in dataset_zero.columns:
                    zero_tk = dataset_zero['acc_tk'].iloc[0]
                    few_tk = dataset_few['acc_tk'].iloc[0]
                    tk_improvement = few_tk - zero_tk
                    tk_improvement_pct = (tk_improvement / zero_tk * 100) if zero_tk != 0 else 0
                    print(f"    TK Accuracy: {zero_tk:.3f} → {few_tk:.3f} ({tk_improvement:+.3f}, {tk_improvement_pct:+.1f}%)")
                
                if ACCURACY_COLUMNS['acc_bs'] and 'acc_bs' in dataset_zero.columns:
                    zero_bs = dataset_zero['acc_bs'].iloc[0]
                    few_bs = dataset_few['acc_bs'].iloc[0]
                    bs_improvement = few_bs - zero_bs
                    bs_improvement_pct = (bs_improvement / zero_bs * 100) if zero_bs != 0 else 0
                    print(f"    BS Accuracy: {zero_bs:.3f} → {few_bs:.3f} ({bs_improvement:+.3f}, {bs_improvement_pct:+.1f}%)")
                
                if ACCURACY_COLUMNS['avg_acc'] and 'avg_acc' in dataset_zero.columns:
                    zero_acc = dataset_zero['avg_acc'].iloc[0]
                    few_acc = dataset_few['avg_acc'].iloc[0]
                    acc_improvement = few_acc - zero_acc
                    acc_improvement_pct = (acc_improvement / zero_acc * 100) if zero_acc != 0 else 0
                    print(f"    Avg Accuracy: {zero_acc:.3f} → {few_acc:.3f} ({acc_improvement:+.3f}, {acc_improvement_pct:+.1f}%)")
                
                # Print performance metrics
                print(f"    Latency: {zero_latency:.2f}s → {few_latency:.2f}s ({latency_change:+.2f}s, {latency_change_pct:+.1f}%)")
                print(f"    Input Cost: ${zero_input_cost:.2f} → ${few_input_cost:.2f} ({input_cost_change:+.2f}, {input_cost_change_pct:+.1f}%)")
                print(f"    Output Cost: ${zero_output_cost:.2f} → ${few_output_cost:.2f} ({output_cost_change:+.2f}, {output_cost_change_pct:+.1f}%)")
                print(f"    Total Cost: ${total_zero_cost:.2f} → ${total_few_cost:.2f} ({total_cost_change:+.2f}, {total_cost_change_pct:+.1f}%)")


print("\n" + "="*80)
print("ALL PERFORMANCE TABLES CREATED SUCCESSFULLY!")
print("="*80)

