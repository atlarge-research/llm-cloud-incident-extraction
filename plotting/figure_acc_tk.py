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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

table_round_mark = 'table_tk'

def load_acc_tk_csv(mode, fields_lst):
    mode = mode.lower()
    df = pd.read_csv(f"{project_root}/results/tables/{table_round_mark}/{table_round_mark}_{mode}.csv")
    fields_lst = [f'{field}_precision' for field in fields_lst] + [f'{field}_recall' for field in fields_lst] + [f'{field}_f1' for field in fields_lst]
    for field in fields_lst:
        df[field] = df[field].apply(lambda x: round(x, 2))
    meta_fields = ['dataset', 'model', 'prompt_type']
    df = df[meta_fields + fields_lst]
    return df

df = load_acc_tk_csv('combined', ['user_symptom_category'])


print(df.head())


# DONE: figures for user_symptom_category
plot_round_mark = 'figure_tk'

# Configure which datasets to plot (configurable)
# Options: 'aws', 'azure', 'gcp', or any combination like ['aws', 'azure'] or ['aws', 'azure', 'gcp']
# Examples:
# DATASETS_TO_PLOT = ['aws']                    # Plot only AWS
# DATASETS_TO_PLOT = ['aws', 'azure']           # Plot AWS and Azure
# DATASETS_TO_PLOT = ['aws', 'azure', 'gcp']    # Plot all three datasets
DATASETS_TO_PLOT = ['aws', 'azure', 'gcp']  # Change this to configure which datasets to plot

# Get available datasets and filter based on configuration
available_datasets = df['dataset'].unique()
available_datasets = [d for d in available_datasets if d != 'dataset']
datasets = [d for d in DATASETS_TO_PLOT if d in available_datasets]

print(f"Available datasets: {available_datasets}")
print(f"Plotting datasets: {datasets}")

# Get unique models
models = df['model'].unique()

print(f"Available models: {list(models)}")
print(f"Number of models: {len(models)}")

# Check if we have datasets to plot
if len(datasets) == 0:
    print("No datasets to plot!")
    exit()

# Define colors for each model series with light/dark variations
model_colors = {
    # GPT series - Greens
    'gpt-3.5': '#90EE90',    # Light green
    'gpt-4o': '#2ca02c',     # Dark green
    
    # Claude series - Oranges  
    'claude-3-5': '#FFB347',  # Light orange (peach)
    'claude-4': '#ff7f0e',    # Dark orange
    
    # Gemini series - Blues
    'gemini-2.0': '#87CEEB',  # Light blue
    'gemini-2.5': '#1f77b4'   # Dark blue
}

# Define hatch patterns for prompt types
hatch_patterns = {0: '', 1: '///'}  # No hatch for 0-shot, diagonal lines for few-shot

# Plotting parameters
x_pos = np.arange(len(models))
width = 0.4

# Create separate plots for each dataset (F1 scores only)
for dataset in datasets:
    # Create a new figure for each dataset
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    # Filter data for current dataset
    dataset_data = df[df['dataset'] == dataset]
    
    # Prepare data for plotting (F1 scores only)
    plot_data = []
    for _, row_data in dataset_data.iterrows():
        plot_data.append({
            'model': row_data['model'],
            'prompt_type': row_data['prompt_type'],
            'f1_score': row_data['user_symptom_category_f1']
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Plot bars for each prompt type
    for j, prompt_type in enumerate([0, 1]):
        data = df_plot[df_plot['prompt_type'] == prompt_type]
        bars = ax.bar(x_pos + j*width, 
                      [data[data['model'] == model]['f1_score'].iloc[0] for model in models],
                      width,
                      label=f'{"Zero-shot" if prompt_type == 0 else "Few-shot"}',
                      color=[model_colors[model] for model in models],
                      hatch=hatch_patterns[prompt_type],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    # Show x-axis tick labels for all subplots
    ax.tick_params(axis='x', which='major', labelsize=12)
    
    # Create appropriate model labels based on actual models
    model_labels = []
    for model in models:
        if 'gpt' in model:
            if '3.5' in model:
                model_labels.append('GPT 3.5')
            elif '4o' in model:
                model_labels.append('GPT 4o')
            else:
                model_labels.append('GPT')
        elif 'claude' in model:
            if '3-5' in model:
                model_labels.append('Claude 3.5')
            elif '4' in model:
                model_labels.append('Claude 4')
            else:
                model_labels.append('Claude')
        elif 'gemini' in model:
            if '2.0' in model:
                model_labels.append('Gemini 2')
            elif '2.5' in model:
                model_labels.append('Gemini 2.5')
            else:
                model_labels.append('Gemini')
        else:
            model_labels.append(model)
    
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels(model_labels, fontsize=10, fontweight='bold', rotation=0, ha='center')
    
    # # Move x-tick labels slightly to the right
    # for label in ax.get_xticklabels():
    #     label.set_horizontalalignment('right')
    #     label.set_x(label.get_position()[0] + 0.5)
    
    # Make all tick labels bold (but keep x-axis at fontsize 10)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='x', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Set title for each subplot
    ax.set_title(f'{dataset.upper()}', fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.9, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(0, 1)
    
    # # Calculate and add average line
    # avg_f1 = df_plot['f1_score'].mean()
    # ax.axhline(y=avg_f1, color='black', linestyle='-', linewidth=2, alpha=0.8, label=f'Avg: {avg_f1:.2f}')
    
    # Add value labels on bars
    for j, prompt_type in enumerate([0, 1]):
        data = df_plot[df_plot['prompt_type'] == prompt_type]
        for k, model in enumerate(models):
            value = data[data['model'] == model]['f1_score'].iloc[0]
            # Adjust x position: move left bar (j=0) slightly left, right bar (j=1) slightly right
            x_offset = -0.05 if j == 0 else 0.05  # Left bar moves left, right bar moves right
            ax.text(k + j*width + x_offset, value + 0.01, f'{value:.2f}', 
                    ha='center', va='bottom', fontsize=10, weight='bold')

    # Create custom legend for this plot
    legend_elements = []
    legend_elements.append(mpatches.Patch(facecolor='white', hatch='', label='Zero-shot', alpha=0.7, edgecolor='black'))
    legend_elements.append(mpatches.Patch(facecolor='white', hatch='///', label='Few-shot', alpha=0.7, edgecolor='black'))

    # Add legend to the current figure with 2 columns (1x2 layout)
    ax.legend(handles=legend_elements, loc='lower center', fontsize=12, prop={'weight': 'bold', 'size': 10}, ncol=2)

    # Adjust layout for this figure
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = f'{project_root}/results/figures/{plot_round_mark}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure for this dataset
    plt.savefig(f'{output_dir}/figure_symp_cat_tk_f1-{dataset}.pdf', 
                bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/figure_symp_cat_tk_f1-{dataset}.png', 
                bbox_inches='tight', dpi=300)
    
    print(f"Figure saved as figure_symp_cat_tk_f1-{dataset}.pdf and .png")
    plt.show()
    plt.close()  # Close the figure to free memory

print("All figures generated successfully!")
