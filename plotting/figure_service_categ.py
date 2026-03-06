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

df_aws['service_category'] = df_aws['service_category'].str.upper()
df_azure['service_category'] = df_azure['service_category'].str.upper()

# all_categories = set(aws_counts.index) | set(azure_counts.index)

aws_counts = df_aws['service_category'].value_counts()
azure_counts = df_azure['service_category'].value_counts()

# Combine
df = pd.DataFrame({"AWS": aws_counts, "Azure": azure_counts}).fillna(0)
# Order by total
df["Total"] = df["AWS"] + df["Azure"]
df = df.sort_values("Total", ascending=True)



# Plot
ax = df[["Azure", "AWS"]].plot(kind="barh", width=0.8, color=[plt.cm.Set1.colors[1], plt.cm.Set1.colors[0]], figsize=(12,4), alpha=0.8)
# plt.title("AWS vs Azure Service Category Counts (Ordered)")
plt.xlabel("Count", fontweight="bold", fontsize=12)
plt.ylabel("Service Category", fontweight="bold", fontsize=12)
# No rotation needed for horizontal bars
ax.tick_params(axis="both", which="major", labelsize=12)
# Make tick labels bold
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
plt.legend(title_fontsize=12, fontsize=12, prop={"weight": "bold"}, ncol=2)

# Add counts at the end of bars
for p in ax.patches:
    ax.annotate(
        str(int(p.get_width())), 
        (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.),
        ha="left", va="center", fontsize=11, rotation=0, fontweight="bold"
    )

# Add grid (vertical lines only)
ax.grid(axis="x", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# save figure pdf and png
os.makedirs(f'{project_root}/results/figures/figure_service_categ', exist_ok=True)
plt.savefig(f'{project_root}/results/figures/figure_service_categ/figure_service_categ_analysis.pdf', format='pdf', dpi=300)
plt.savefig(f'{project_root}/results/figures/figure_service_categ/figure_service_categ_analysis.png', format='png', dpi=300)
plt.show()

# Plot percentage

# Calculate percentages relative to each provider's total
df["aws_percent"] = df["AWS"] / df["AWS"].sum() * 100
df["azure_percent"] = df["Azure"] / df["Azure"].sum() * 100

# Order by total count
df["Total"] = df["AWS"] + df["Azure"]
df = df.sort_values("Total", ascending=True)

# Plot percentage
ax = df[["azure_percent", "aws_percent"]].plot(kind="barh", width=0.8, color=[plt.cm.Set1.colors[1], plt.cm.Set1.colors[0]], figsize=(6,6), alpha=0.8)
plt.xlabel("Percentage", fontweight="bold", fontsize=12)
plt.ylabel("")
# No rotation needed for horizontal bars
ax.tick_params(axis="both", which="major", labelsize=12)
# Make tick labels bold
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
# the legend is AWS and Azure
legend = plt.legend(labels=["AWS", "AZURE"], title_fontsize=12, fontsize=14, prop={"weight": "bold"}, ncol=1)
legend.legend_handles[0].set_color(plt.cm.Set1.colors[0])  # AWS = red
legend.legend_handles[1].set_color(plt.cm.Set1.colors[1])  # Azure = blue


# Add percentage at the end of bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_width():.2f}%", 
        (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.),
        ha="left", va="center", fontsize=12, rotation=0, fontweight="bold"
    )

# Add grid (vertical lines only)
ax.grid(axis="x", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# save figure pdf and png
os.makedirs(f'{project_root}/results/figures/figure_service_categ', exist_ok=True)
plt.savefig(f'{project_root}/results/figures/figure_service_categ/figure_service_categ_analysis_percent.pdf', format='pdf', dpi=300)
plt.savefig(f'{project_root}/results/figures/figure_service_categ/figure_service_categ_analysis_percent.png', format='png', dpi=300)
plt.show()