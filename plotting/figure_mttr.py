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

from utils.io import load_extraction_jsonl, load_extraction_for_analysis_jsonl



df_aws = load_extraction_for_analysis_jsonl(round_time="anl", dataset="aws", model_abbr="gemini-2.5", model_name="gemini-2.5-pro", prompt_type="1")
df_azure = load_extraction_for_analysis_jsonl(round_time="anl", dataset="azure", model_abbr="gemini-2.0", model_name="gemini-2.0-flash", prompt_type="1")

def process_mttr_data(df, dataset_name):
    """Process MTTR data for a given dataframe"""
    # handle 'UNKNOWN' values in start_time and end_time
    df['start_time'] = df['start_time'].replace('UNKNOWN', np.nan)
    df['end_time'] = df['end_time'].replace('UNKNOWN', np.nan)
    
    # transform start_time and end_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # calculate mttr
    df['mttr'] = df['end_time'] - df['start_time']
    # transform mttr to hours
    df['mttr'] = df['mttr'] / np.timedelta64(1, 'h')

    # select mttr larger than 0
    df = df[df['mttr'] > 0]
    
    # exclude outlier values in mttr larger than 3 sigma
    df = df[df['mttr'] < df['mttr'].mean() + 3 * df['mttr'].std()]
    
    # add dataset label
    df['dataset'] = dataset_name
    
    return df

# Process both datasets
df_aws_processed = process_mttr_data(df_aws.copy(), 'AWS')
df_azure_processed = process_mttr_data(df_azure.copy(), 'AZURE')

# Combine datasets for plotting
df_combined = pd.concat([df_aws_processed, df_azure_processed], ignore_index=True)

# plot box plot for mttr comparison
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
sns.boxplot(data=df_combined, x='mttr', y='dataset', orient='h', width=0.6, 
            palette='Set1', linecolor='black')
ax.set_xlabel(r'MTTR [hours]', fontsize=18, fontweight='bold')
ax.set_ylabel('')
# ax.set_title('Mean Time To Recovery (MTTR) Comparison: AWS vs Azure', fontsize=16, fontweight='bold')
ax.grid(axis='both', linestyle='--', alpha=0.6, which='both')
ax.set_xscale('log')
# Make tick labels bold and larger
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=16)
# Set tick label font weight to bold
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
ax.set_xlim(0, 100)

# add reference lines for 30 minutes, 1 hour, 3 hours, 10 hours, 24 hours
ax.axvline(x=30/60, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(30/60, -0.5, r'0.5h', color='black', fontsize=18, fontweight='bold', ha='center', verticalalignment='top')
ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(1, -0.5, r'1h', color='black', fontsize=18, fontweight='bold', ha='center', verticalalignment='top')
ax.axvline(x=3, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(3, -0.5, r'3h', color='black', fontsize=18, fontweight='bold', ha='center', verticalalignment='top')
ax.axvline(x=10, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(10, -0.5, r'10h', color='black', fontsize=18, fontweight='bold', ha='center', verticalalignment='top')
ax.axvline(x=24, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(24, -0.5, r'24h', color='black', fontsize=18, fontweight='bold', ha='center', verticalalignment='top')

# add median value
median_values = df_combined.groupby('dataset')['mttr'].median()
for dataset, median in median_values.items():
    ax.text(median, dataset, f'{median:.2f}h', color='black', fontsize=18, ha='center', verticalalignment='center', fontweight='bold')
    # ax.text(median, dataset, f'{median:.2f}h', color='black', fontsize=16, fontweight='bold', ha='center', verticalalignment='center')


plt.tight_layout()
# save figure pdf and png
# make directory if not exists
os.makedirs(f'{project_root}/results/figures/figure_mttr', exist_ok=True)
plt.savefig(f'{project_root}/results/figures/figure_mttr/figure_mttr_analysis.pdf', format='pdf', dpi=300)
plt.savefig(f'{project_root}/results/figures/figure_mttr/figure_mttr_analysis.png', format='png', dpi=300)
plt.show()