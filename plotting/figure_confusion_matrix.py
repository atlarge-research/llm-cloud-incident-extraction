import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import os
import sys
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.insert(0, project_root)
import yaml

from utils.io import load_extraction_jsonl, load_label_csv

# Configuration parameters
# field_name_lst = ['service_category', 'root_cause_category']
field_name_lst = ['root_cause_category']
service_category_lst = ['COMPUTE', 'STORAGE', 'NETWORK', 'SECURITY', 'AI', 'MANAGEMENT', 'ANALYTICS', 'DATABASE', 'OTHERS']
root_cause_lst = ['CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN']

# Compute confusion matrix for each dataset, model, and prompt type
# dataset_lst = ['aws', 'azure', 'gcp']
with open(f'{project_root}/config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)
model_dict = model_config['models']
model_abbr_lst = list(model_dict.keys())
model_name_lst = list(model_dict.values())
prompt_type_lst = ['0', '1']

extraction_round_time = 'ext'

def clean_service_category(category):
    valid = service_category_lst
    if not isinstance(category, str):
        return 'OTHERS'
    return category if category in valid else 'OTHERS'

def clean_root_cause(category):
    valid = root_cause_lst
    if not isinstance(category, str):
        return 'OTHERS'
    return category if category in valid else 'OTHERS'

def load_and_clean_data(dataset, model_abbr, model_name, prompt_type, extraction_round_time):
    """Load and clean extraction data for a specific configuration"""
    try:
        df_extract = load_extraction_jsonl(extraction_round_time, dataset, model_abbr, model_name, prompt_type)
        
        # Check what columns are actually available
        available_columns = df_extract.columns.tolist()
        print(f"Available columns in dataset: {available_columns}")
        
        # Map expected field names to actual column names (case-insensitive)
        field_mapping = {}
        for expected_field in ['service_category', 'root_cause_category']:
            # Try exact match first
            if expected_field in available_columns:
                field_mapping[expected_field] = expected_field
            else:
                # Try case-insensitive match
                found = False
                for col in available_columns:
                    if col.lower() == expected_field.lower():
                        field_mapping[expected_field] = col
                        found = True
                        break
                if not found:
                    print(f"Warning: Column '{expected_field}' not found in dataset")
                    field_mapping[expected_field] = None
        
        # Clean the data only if columns exist
        if field_mapping['service_category']:
            df_extract['service_category'] = df_extract[field_mapping['service_category']].apply(clean_service_category)
        else:
            print("Warning: service_category column not available, skipping...")
            
        if field_mapping['root_cause_category']:
            df_extract['root_cause_category'] = df_extract[field_mapping['root_cause_category']].apply(clean_root_cause)
        else:
            print("Warning: root_cause_category column not available, skipping...")
        
        return df_extract
    except Exception as e:
        print(f"Error loading data for {dataset}/{model_abbr}-{prompt_type}: {e}")
        return None

def load_and_clean_labels(dataset):
    """Load and clean label data for a specific dataset"""
    try:
        df_label = load_label_csv(dataset)
        
        # Check what label columns are actually available
        available_label_columns = df_label.columns.tolist()
        print(f"Available label columns: {available_label_columns}")
        
        # Map expected label field names to actual column names (case-insensitive)
        label_field_mapping = {}
        for expected_label_field in ['label_service_category', 'label_root_cause_category']:
            # Try exact match first
            if expected_label_field in available_label_columns:
                label_field_mapping[expected_label_field] = expected_label_field
            else:
                # Try case-insensitive match
                found = False
                for col in available_label_columns:
                    if col.lower() == expected_label_field.lower():
                        label_field_mapping[expected_label_field] = col
                        found = True
                        break
                if not found:
                    print(f"Warning: Label column '{expected_label_field}' not found in dataset")
                    label_field_mapping[expected_label_field] = None
        
        # Clean label data only if columns exist
        for expected_col, actual_col in label_field_mapping.items():
            if actual_col:
                df_label[expected_col] = df_label[actual_col].apply(lambda x: x.upper())
            else:
                # Create empty column if not found
                df_label[expected_col] = 'UNKNOWN'
        
        return df_label
    except Exception as e:
        print(f"Error loading labels for {dataset}: {e}")
        return None

def compute_confusion_matrix(df_extract, df_label, field_name):
    """Compute confusion matrix for a specific field"""
    try:
        # Check if the required columns exist
        if field_name not in df_extract.columns:
            print(f"Error: Field '{field_name}' not found in extraction data")
            return None
            
        label_col = f'label_{field_name}'
        if label_col not in df_label.columns:
            print(f"Error: Label column '{label_col}' not found in label data")
            return None
        
        df_cm = pd.crosstab(df_extract[field_name], df_label[label_col], 
                           rownames=['Predicted'], colnames=['Actual'])
        return df_cm
    except Exception as e:
        print(f"Error computing confusion matrix for {field_name}: {e}")
        return None

def plot_single_confusion_matrix(cm, field_name, normalize="row", save_path= '{project_root}/results/figures/confusion_matrix', dataset=None, model_abbr=None, prompt_type=None):
    """Plot a single confusion matrix"""
    if field_name == 'service_category':
        category_labels = service_category_lst
    elif field_name == 'root_cause_category':
        category_labels = root_cause_lst
    else:
        print(f"Unknown field: {field_name}")
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Reindex to ensure all categories are present
    cm = cm.reindex(index=category_labels, columns=category_labels, fill_value=0)
    
    if normalize == "total":
        cm_values = cm / cm.values.sum() * 100
        title_suffix = " (% of total)"
    elif normalize == "row":
        cm_values = cm.div(cm.sum(axis=1), axis=0) * 100
        title_suffix = " (% of predicted)"
    elif normalize == "column":
        cm_values = cm.div(cm.sum(axis=0), axis=1) * 100
        title_suffix = " (% of actual)"
    elif normalize == "none":
        cm_values = cm
        title_suffix = " (counts)"
    
    # Create discrete color bands by binning the data
    import matplotlib.colors as mcolors
    
    # Bin the data into discrete ranges
    cm_binned = cm_values.copy()
    cm_binned[cm_binned <= 20] = 10  # 0-20 range
    cm_binned[(cm_binned > 20) & (cm_binned <= 40)] = 30  # 20-40 range
    cm_binned[(cm_binned > 40) & (cm_binned <= 60)] = 50  # 40-60 range
    cm_binned[(cm_binned > 60) & (cm_binned <= 80)] = 70  # 60-80 range
    cm_binned[cm_binned > 80] = 90  # 80-100 range
    
    # Create discrete colormap with light grey to dark grey
    colors = ['#f5f5f5', '#d3d3d3', '#a9a9a9', '#696969', '#2f2f2f']  # Light grey to dark grey
    n_bins = 5
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 20, 40, 60, 80, 100]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    sns.heatmap(cm_binned, annot=cm_values, fmt=".2f", cmap=cmap, norm=norm,
                xticklabels=category_labels, yticklabels=category_labels,
                ax=ax, cbar=True, annot_kws={'weight': 'bold', 'size': 10},
                vmin=0, vmax=100, square=True)
    
    ax.set_xlabel("Actual", fontweight='bold', fontsize=14)
    ax.set_ylabel("Predicted", fontweight='bold', fontsize=14)
    # Create title with dataset-model-prompt type format
    if dataset and model_abbr and prompt_type is not None:
        if prompt_type == '0':
            prompt_type = 'Zero-shot'
        elif prompt_type == '1':
            prompt_type = 'Few-shot'
        # title = f"{dataset.upper()} - {model_abbr.upper()} - {prompt_type}"
        title = f"{dataset.upper()} {model_abbr.upper()}"
    else:
        title = f"Confusion Matrix - {field_name.replace('_', ' ').title()}{title_suffix}"
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    # Rotate x-axis labels for better readability and make them bold
    ax.set_xticklabels(category_labels, rotation=45, ha='right', fontweight='bold', fontsize=12)
    ax.set_yticklabels(category_labels, rotation=0, fontweight='bold', fontsize=12)
    
    # Make colorbar labels bold and larger
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved confusion matrix plot to: {save_path}")
    
    plt.show()
    return fig

def get_available_fields(df_extract, df_label):
    """Dynamically determine which fields are available in the data"""
    available_fields = []
    
    # Check which fields exist in extraction data
    for field in field_name_lst:
        if field in df_extract.columns and f'label_{field}' in df_label.columns:
            available_fields.append(field)
        else:
            print(f"Field '{field}' not available (missing from extraction data or labels)")
    
    return available_fields

def generate_confusion_matrices(selected_datasets=None, selected_apis=None, selected_models=None, 
                               selected_prompt_types=None, selected_fields=None, normalize="row", 
                               save_plots=True, output_dir=f'{project_root}/results/figures/confusion_matrix'):
    """Generate confusion matrices for all specified configurations"""
    
    # Use all if none specified
    if selected_datasets is None:
        selected_datasets = dataset_lst
    if selected_apis is None:
        selected_apis = list(model_dict.keys())
    if selected_models is None:
        selected_models = model_name_lst
    if selected_prompt_types is None:
        selected_prompt_types = prompt_type_lst
    if selected_fields is None:
        selected_fields = field_name_lst
    
    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for dataset in selected_datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*50}")
        
        # Load labels once per dataset
        df_label = load_and_clean_labels(dataset)
        if df_label is None:
            continue
            
        results[dataset] = {}
        
        for api in selected_apis:
            results[dataset][api] = {}
            
            for model in selected_models:
                results[dataset][api][model] = {}
                
                for prompt_type in selected_prompt_types:
                    print(f"\nProcessing: {dataset}-{api}-{model}-{prompt_type}")
                    
                    # Load extraction data
                    df_extract = load_and_clean_data(dataset, api, model, prompt_type, extraction_round_time)
                    if df_extract is None:
                        continue
                    
                    # Determine which fields are actually available for this configuration
                    available_fields = get_available_fields(df_extract, df_label)
                    if not available_fields:
                        print(f"  No available fields found for {dataset}-{api}-{model}-{prompt_type}")
                        continue
                    
                    # Filter selected fields to only include available ones
                    fields_to_process = [f for f in selected_fields if f in available_fields]
                    if not fields_to_process:
                        print(f"  None of the selected fields are available for {dataset}-{api}-{model}-{prompt_type}")
                        continue
                    
                    results[dataset][api][model][prompt_type] = {}
                    
                    for field in fields_to_process:
                        print(f"{'='*50}")
                        print(f"Computing confusion matrix for {field}...")
                        
                        # Compute confusion matrix
                        cm = compute_confusion_matrix(df_extract, df_label, field)
                        if cm is None:
                            continue
                        
                        results[dataset][api][model][prompt_type][field] = cm
                        
                        # Plot confusion matrix
                        if save_plots:
                            save_path = os.path.join(output_dir, f"figure-heatmap-{dataset}-{api}-{prompt_type}.pdf")
                        else:
                            save_path = None
                        
                        plot_single_confusion_matrix(cm, field, normalize, save_path, dataset, api, prompt_type)
                        
                        # Print summary statistics
                        print(f"Confusion matrix shape: {cm.shape}")
                        print(f"Total samples: {cm.values.sum()}")
    
    return results

def print_confusion_matrix_summary(results):
    """Print a summary of all confusion matrices"""
    print("\n" + "="*80)
    print("CONFUSION MATRIX SUMMARY")
    print("="*80)
    
    for dataset, dataset_results in results.items():
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        
        for api, api_results in dataset_results.items():
            for model, model_results in api_results.items():
                for prompt_type, prompt_results in model_results.items():
                    print(f"\n  {api} - {model} - Prompt {prompt_type}:")
                    
                    for field, cm in prompt_results.items():
                        accuracy = np.trace(cm) / cm.values.sum() * 100
                        print(f"    {field}: {accuracy:.2f}% accuracy")

# Example usage functions
def generate_all_confusion_matrices(normalize="row", save_plots=False):
    """Generate confusion matrices for all configurations"""
    return generate_confusion_matrices(
        normalize=normalize,
        save_plots=save_plots
    )

def generate_specific_confusion_matrices(datasets, model_abbr, model_name, 
                                       prompt_types, fields, normalize="row"):
    """Generate confusion matrices for specific configurations"""
    return generate_confusion_matrices(
        selected_datasets=datasets,
        selected_apis=model_abbr,
        selected_models=model_name,
        selected_prompt_types=prompt_types,
        selected_fields=fields,
        normalize=normalize
    )


# Main execution
if __name__ == "__main__":
    # # Example: Generate confusion matrices for all configurations
    # print("Generating confusion matrices for all configurations...")
    # results = generate_all_confusion_matrices(normalize="row", save_plots=False)
    # # Print summary
    # print_confusion_matrix_summary(results)
    
    ## Example: Generate specific confusion matrices
    print("\nGenerating specific confusion matrices...")
    specific_results = generate_specific_confusion_matrices(
        datasets=['azure'],
        model_abbr=['gemini-2.0'],
        model_name=['gemini-2.0-flash'],
        prompt_types=['1'],
        fields=['root_cause_category']
    )

