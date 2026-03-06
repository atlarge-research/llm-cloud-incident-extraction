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


table_round_mark = 'table_bs'

def load_acc_bs_csv(mode, fields_lst):
    mode = mode.lower()
    df = pd.read_csv(f"{project_root}/results/tables/{table_round_mark}/{table_round_mark}_{mode}.csv")
    fields_lst = [f'{field}_precision' for field in fields_lst] + [f'{field}_recall' for field in fields_lst] + [f'{field}_f1' for field in fields_lst]
    # for field in fields_lst:
    #     df[field] = df[field].apply(lambda x: round(x / 100, 2))
    meta_fields = ['dataset', 'model', 'prompt_type']
    df = df[meta_fields + fields_lst]
    return df



# Load data for root_cause (plot: azure)
df_root_cause = load_acc_bs_csv('combined', ['root_cause'])


plot_round_mark = 'figure_bs'
# Create line chart figures for BERT scores
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def filter_valid_datasets(df, field_name):
    """Filter datasets that have at least one non-NaN value for the given field"""
    valid_datasets = []
    for dataset in df['dataset'].unique():
        if dataset == 'dataset':  # Skip header row
            continue
        dataset_data = df[df['dataset'] == dataset]
        # Check if there's at least one non-NaN value for the field
        if not dataset_data[field_name].isna().all():
            valid_datasets.append(dataset)
    return valid_datasets




# Get unique models
models = df_root_cause['model'].unique()

# Define colors for prompt types
prompt_colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue for zero-shot, Orange for few-shot

# Define markers for models
model_markers = {
    'gpt-3.5': 'o',
    'gpt-4o': 's',
    'claude-3-5': '^',
    'claude-4': 'D',
    'gemini-2.0': 'v',
    'gemini-2.5': 'X'
}


# Create figure for root_cause
n_root_cause = 1
if n_root_cause > 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))    
    # Plot root_cause BERT scores
    for col, dataset in enumerate(['azure']):
        
        # Filter data for current dataset
        dataset_data = df_root_cause[df_root_cause['dataset'] == dataset]
        
        # Plot lines for each prompt type
        for prompt_type in [0, 1]:
            df_prompt = dataset_data[dataset_data['prompt_type'] == prompt_type]
            
            # Filter out NaN values
            df_prompt = df_prompt.dropna(subset=['root_cause_f1'])
            
            # Skip if no valid data for this prompt type
            if df_prompt.empty:
                continue
                
            # Ensure models appear in fixed order
            df_prompt = df_prompt.set_index('model').loc[model_markers.keys()].reset_index()
            
            # Plot line
            ax.plot(
                df_prompt['model'], 
                df_prompt['root_cause_f1'], 
                label=f'{"Zero-shot" if prompt_type == 0 else "Few-shot"}',
                color=prompt_colors[prompt_type],
                linestyle='-',
                linewidth=2
            )
            
            # Add markers by model
            for m, y in zip(df_prompt['model'], df_prompt['root_cause_f1']):
                ax.scatter(m, y, marker=model_markers[m], color=prompt_colors[prompt_type], s=80)
                
                # Add value labels: above for few-shot, below for zero-shot
                if prompt_type == 1:  # few-shot
                    ax.text(m, y + 0.02, f'{y:.2f}', 
                           ha='center', va='bottom', fontsize=20, weight='bold')
                else:  # zero-shot
                    ax.text(m, y - 0.02, f'{y:.2f}', 
                           ha='center', va='top', fontsize=20, weight='bold')
        
        # Customize the subplot
        ax.set_xlabel('Models', fontsize=20, fontweight='bold')
        
        # Only add y-label for the leftmost subplot
        if col == 0:
            ax.set_ylabel('BERTScore', fontsize=20, fontweight='bold')
            ax.tick_params(axis='y', which='major', labelsize=18)
        else:
            # Hide y-axis labels for other columns
            ax.tick_params(axis='y', which='major', labelsize=18, labelleft=True)
        
        # Show x-axis tick labels for all subplots
        ax.tick_params(axis='x', which='major', labelsize=18)
        
        # Create appropriate model labels based on actual models
        models = list(model_markers.keys())  # Get the 6 models in order
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
        
        ax.set_xticklabels(model_labels, fontsize=15, fontweight='bold', rotation=0, ha='center')
        
        # Make all tick labels bold
        ax.tick_params(axis='y', which='major', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Set title for each subplot
        ax.set_title(f'{dataset.upper()}', fontsize=22, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.7, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set y-axis to start from 0.4 for better comparison
        ax.set_ylim(0.4, 1)
        
        # Add legend for each subplot in the lower center
        legend_elements_root = []
        for prompt_type in [1, 0]:
            legend_elements_root.append(plt.Line2D([0], [0], 
                                             color=prompt_colors[prompt_type], 
                                             linewidth=2,
                                             label=f'{"Zero-shot" if prompt_type == 0 else "Few-shot"}'))
        
        ax.legend(handles=legend_elements_root, loc='lower center', 
                 fontsize=20, prop={'weight': 'bold', 'size': 16})
    
    # Create custom legend for root_cause
    # legend_elements_root = []
    # for prompt_type in [1, 0]:
    #     legend_elements_root.append(plt.Line2D([0], [0], 
    #                                          color=prompt_colors[prompt_type], 
    #                                          linewidth=2,
    #                                          label=f'{"Zero-shot" if prompt_type == 0 else "Few-shot"}'))
    
    # Add legend to the top center of the figure
    # fig2.legend(handles=legend_elements_root, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
    #            fontsize=20, ncol=1, prop={'weight': 'bold'})
    
    # plt.suptitle('BERT Score (F1) for Root Cause Extraction', fontsize=24, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = f'{project_root}/results/figures/{plot_round_mark}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the root_cause figure
    plt.savefig(f'{output_dir}/figure_root_cause_bs_azure.pdf', 
                bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/figure_root_cause_bs_azure.png', 
                bbox_inches='tight', dpi=300)
    
    print("Root Cause BERT Score figure saved as figure_root_cause_bs_azure.pdf and .png")
    plt.show()



