import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import os
import sys
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.insert(0, project_root)
import yaml


# load table_perf_cost.csv
df = pd.read_csv(f'{project_root}/results/tables/table_perf_cost/table_perf_cost.csv')

print(df.columns)

# set parameters
with open(f"../config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    model_dict = config['models']
model_abbr_lst = list(model_dict.keys())
dataset_lst = ['aws', 'azure', 'gcp']
prompt_type_lst = ['0', '1']
mode = 'single' # 'combined' or 'single'


def create_bubble_plot(data, save_name, figsize=(6, 8), title=None):
    """Create and save a bubble plot"""
    plt.figure(figsize=figsize)
    
    # Create scatter plot with different shapes for different model series
    # Define shape mapping for model series
    model_series_shapes = {
        'gpt': 's',          # square for GPT series
        'claude': 'o',       # circle for Claude series
        'gemini': '^',       # triangle up for Gemini series
    }
    
    # Define advanced models (will use hatched patterns)
    advanced_models = ['gpt-4o', 'claude-4', 'gemini-2.5']
    
    # Define color mapping for prompt types
    prompt_type_colors = {
        '0': 'blue',         # 0-shot
        '1': 'orange',       # few-shot
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    
    # Plot each model with different shapes and colors
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        
        # Determine model series for shape
        if 'gpt' in model:
            series = 'gpt'
        elif 'claude' in model:
            series = 'claude'
        elif 'gemini' in model:
            series = 'gemini'
        else:
            series = 'gpt'  # default
        
        shape = model_series_shapes.get(series, 'o')
        
        # Color by prompt type
        for prompt_type in model_data['prompt_type'].unique():
            prompt_data = model_data[model_data['prompt_type'] == prompt_type]
            # Try both string and int keys
            color = prompt_type_colors.get(str(prompt_type), prompt_type_colors.get(int(prompt_type), 'blue'))
            
            # Determine if this is an advanced model
            is_advanced = model in advanced_models
            
            if is_advanced:
                # Advanced model: filled shape with color
                ax.scatter(
                    prompt_data['avg_total_cost_10m4'],
                    prompt_data['avg_acc'],
                    c=color,
                    marker=shape,
                    s=400,  # increased size
                    # alpha=0.7,
                    edgecolors=color,
                    linewidth=4,
                    label=f"{model}_{prompt_type}" if len(data['prompt_type'].unique()) > 1 else model
                )
            else:
                # Basic model: unfilled shape with prompt color edge
                ax.scatter(
                    prompt_data['avg_total_cost_10m4'],
                    prompt_data['avg_acc'],
                    c='white',
                    marker=shape,
                    s=400,  # increased size
                    alpha=0.7,
                    edgecolors=color,
                    linewidth=4,
                    label=f"{model}_{prompt_type}" if len(data['prompt_type'].unique()) > 1 else model
                )
    
    # Add annotations for each point
    for idx, row in data.iterrows():
        # Format model name for display
        model_abbr = row['model']
        prompt_type = row['prompt_type']
        
        # Convert model abbreviation to display name
        if model_abbr == 'gpt-3.5':
            display_model = 'GPT3.5'
        elif model_abbr == 'gpt-4o':
            display_model = 'GPT4o'
        elif model_abbr == 'claude-3-5':
            display_model = 'Claude3.5'
        elif model_abbr == 'claude-4':
            display_model = 'Claude4'
        elif model_abbr == 'gemini-2.0':
            display_model = 'Gemini2'
        elif model_abbr == 'gemini-2.5':
            display_model = 'Gemini2.5'
        else:
            display_model = model_abbr.title()
        
        # Create annotation text
        annotation_text = f"{display_model}\n{prompt_type}"
        
        # # Add annotation with slight offset to avoid overlapping with points
        # plt.annotate(
        #     annotation_text,
        #     xy=(row['avg_total_cost_10m4'] + 0.01, row['avg_acc']),
        #     xytext=(-30, 10),
        #     textcoords='offset points',
        #     fontsize=12,
        #     fontweight='bold',
        #     alpha=0.9,
        #     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.4, edgecolor='gray')
        # )
    
    # Customize the plot
    plt.xlabel(f'Cost', fontsize=19, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=19, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold') if title else None
    
    # Set x-axis to log scale
    # plt.xscale('log')
    
    # Make axis ticks bold and larger
    plt.xticks(fontsize=18, fontweight='bold')  # Increased from 14 to 18
    plt.yticks(fontsize=18, fontweight='bold')  # Increased from 14 to 18
    
    # Format y-axis ticks to 2 decimal places
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Customize legend
    # Create legend for prompt types (colors)
    prompt_handles = []
    prompt_labels = []
    
    for prompt_type, color in prompt_type_colors.items():
        prompt_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12))
        if prompt_type == '0':
            prompt_labels.append("zero-shot")
        elif prompt_type == '1':
            prompt_labels.append("few-shot")
        else:
            prompt_labels.append(f"{prompt_type}-shot")
    
    # Create legend for models (shapes)
    model_handles = []
    model_labels = []
    
    # Define model display names
    model_display_names = {
        'gpt-3.5': 'GPT3.5',
        'gpt-4o': 'GPT4o',
        'claude-3-5': 'Claude3.5',
        'claude-4': 'Claude4',
        'gemini-2.0': 'Gemini2.0',
        'gemini-2.5': 'Gemini2.5'
    }
    
    # Get unique models from data
    unique_models = data['model'].unique()
    
    for model in unique_models:
        # Determine model series for shape
        if 'gpt' in model:
            series = 'gpt'
        elif 'claude' in model:
            series = 'claude'
        elif 'gemini' in model:
            series = 'gemini'
        else:
            series = 'gpt'  # default
        
        shape = model_series_shapes.get(series, 'o')
        is_advanced = model in advanced_models
        display_name = model_display_names.get(model, model.upper())
        
        if is_advanced:
            # Advanced model (filled with black)
            model_handles.append(plt.Line2D([0], [0], marker=shape, color='black', markersize=12, linestyle='None', 
                                            markerfacecolor='black', markeredgecolor='black', markeredgewidth=4))
            model_labels.append(display_name)
        else:
            # Basic model (unfilled with black edge)
            model_handles.append(plt.Line2D([0], [0], marker=shape, color='black', markersize=12, linestyle='None', 
                                            markerfacecolor='white', markeredgecolor='black', markeredgewidth=4))
            model_labels.append(display_name)
    
    # Combine legends
    all_handles = prompt_handles + model_handles
    all_labels = prompt_labels + model_labels
    
    plt.legend(all_handles, all_labels, title_fontsize=18, fontsize=16, 
              bbox_to_anchor=(1, 0), loc='lower right')
    
    # Add average lines
    avg_cost = data['avg_total_cost_10m4'].mean()
    avg_acc = data['avg_acc'].mean()
    
    # Vertical line for average cost
    plt.axvline(x=avg_cost, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Horizontal line for average accuracy
    plt.axhline(y=avg_acc, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add annotations for average values
    # Annotation for average cost (to the right of the vertical line, lower position)
    plt.annotate(f'Avg. Cost: {avg_cost:.2f}', 
                xy=(avg_cost, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.8), 
                xytext=(2, 0), 
                textcoords='offset points',
                fontsize=14, fontweight='bold', color='red')
    
    # Annotation for average accuracy (above the horizontal line, more to the left)
    plt.annotate(f'Avg. Acc.: {avg_acc:.2f}', 
                xy=(plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.7, avg_acc - 0.01), 
                xytext=(0, 2), 
                textcoords='offset points',
                fontsize=14, fontweight='bold', color='green')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(f'{project_root}/results/figures/figure_bubble', exist_ok=True)
    plt.savefig(f'{project_root}/results/figures/figure_bubble/{save_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{project_root}/results/figures/figure_bubble/{save_name}.png', dpi=300, bbox_inches='tight')
    print(f"Bubble plot saved: {save_name}")
    
    # Show the plot
    plt.show()

# Create plots based on mode
if mode == 'combined':
    # Combined plot with all datasets
    create_bubble_plot(
        data=df,
        # title='Performance vs Cost vs Latency - All Datasets',
        save_name='figure_bubble_combined'
    )
    
elif mode == 'single':
    # Individual plots for each dataset
    for dataset in dataset_lst:
        dataset_data = df[df['dataset'] == dataset]
        if not dataset_data.empty:
            create_bubble_plot(
                data=dataset_data,
                title=f'{dataset.upper()}',
                save_name=f'figure_bubble_{dataset}',
                figsize=(7, 7)
            )
        else:
            print(f"No data found for dataset: {dataset}")
            
else:
    print(f"Invalid mode: {mode}. Please use 'combined' or 'single'")