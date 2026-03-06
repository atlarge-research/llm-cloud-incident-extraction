import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from plotnine import ggplot
import string
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

import os
import sys
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.insert(0, project_root)
print(project_root)


dfs = {}
path = "llm-data-extraction/data/2_clean_data"
cloud = ["aws", "azure", "gcp"]
for c in cloud:
    dfs[c] = pd.read_csv(f"{project_root}/{path}/{c}.csv")

dfs_sample = {}
sample_path = "llm-data-extraction/data/3_sample_data"
for c in cloud:
    dfs_sample[c] = pd.read_csv(f"{project_root}/{sample_path}/{c}_sample.csv")

# transform ['description'] html to text for all dataframes
def clean_html_text(text):
    """Convert HTML to plain text"""
    return BeautifulSoup(text, 'html.parser').get_text()

for cloud_provider in dfs:
    dfs[cloud_provider]['description'] = dfs[cloud_provider]['description'].apply(clean_html_text)

# calculate the number of rows in dfs
for cloud_provider in dfs:
    print(f"{cloud_provider}: {len(dfs[cloud_provider])}")

# calculate the number of rows in dfs_sample
for cloud_provider in dfs_sample:
    print(f"{cloud_provider} (label): {len(dfs_sample[cloud_provider])}")


    # calculate the average length of the description

def count_words(text):
    """Count words in text after removing punctuation"""
    return len(
        str(text)
        .translate(str.maketrans('', '', string.punctuation))
        .split()
    )


# Calculate word counts and averages for all cloud providers
word_counts = {}
averages = {}

for cloud_provider in dfs:
    word_counts[cloud_provider] = dfs[cloud_provider]['description'].apply(count_words)
    averages[cloud_provider] = np.mean(word_counts[cloud_provider])
    print(f"{cloud_provider.upper()} length: {averages[cloud_provider]} words")