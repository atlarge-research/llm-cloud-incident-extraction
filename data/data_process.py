import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from datetime import timedelta, datetime
from pprint import pprint
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from bs4 import BeautifulSoup

'''
## Start: Raw data for structured data extraction

- 1_raw_data
- 2_clean_data
- 3_sample_data
- 4_label_data
'''
# Load raw data
operator_lst = ['aws', 'azure', 'gcp']
dfs = {}
for operator in operator_lst:
    df = pd.read_parquet(f'../data/1_raw_data/{operator}_provider.parquet')
    dfs[operator] = df

print('aws:\n', len(dfs['aws']), dfs['aws'].columns)
display(dfs['aws'].head(5))
print('azure:\n', len(dfs['azure']), dfs['azure'].columns)
display(dfs['azure'].head(5))
print('gcp:\n', len(dfs['gcp']), dfs['gcp'].columns)
display(dfs['gcp'].head(5))


'''
## Clean data
- Select useful columns.
- Unify the columns names for different operators. 
- Save only description columns for sampling selection.
'''

# select columns
dfs['aws'] = dfs['aws'][['date', 'service_name', 'summary', 'description', 'year', 'vendor']]
dfs['gcp'] = dfs['gcp'][['event_start_time', 'external_desc', 'service_name', 'updates', 'vendor']]
dfs['gcp']['updates'] = dfs['gcp']['updates'].astype(str)

# rename and reorder columns
dfs['aws'] = dfs['aws'].rename(columns={'service_name': 'service_name', 'summary': 'user_symptoms', 'description': 'description', 'year': 'year', 'vendor': 'vendor'})
dfs['aws'] = dfs['aws'][['service_name', 'description', 'user_symptoms', 'vendor', 'year']]

dfs['gcp'] = dfs['gcp'].rename(columns={'service_name': 'service_name', 'external_desc': 'external_description', 'updates': 'description', 'vendor': 'vendor'})
dfs['gcp'] = dfs['gcp'][['service_name', 'description', 'external_description', 'vendor']]

# TODO: convert HTML to text

dfs['azure']['vendor'] = 'Azure'  # add column vendor == Azure

display(dfs['aws'].head(5))
display(dfs['azure'].head(5))
display(dfs['gcp'].head(5))


# save cleaned data as csv
dfs['aws'].to_csv('../data/2_clean_data/aws.csv', index=False)
dfs['gcp'].to_csv('../data/2_clean_data/gcp.csv', index=False)
dfs['azure'].to_csv('../data/2_clean_data/azure.csv', index=False)

# save only description column cleaned data as csv
dfs['aws'][['description']].to_csv('../data/2_clean_data/aws_description.csv', index=False)
dfs['gcp'][['description']].to_csv('../data/2_clean_data/gcp_description.csv', index=False)
dfs['azure'][['description']].to_csv('../data/2_clean_data/azure_description.csv', index=False)


'''
## Sample data: Use K-means to get sample data for annotation 
'''

# Load clean data
operator_lst = ['aws', 'azure', 'gcp']
dfs = {}
for operator in operator_lst:
    df = pd.read_csv(f'../data/2_clean_data/{operator}_description.csv')
    dfs[operator] = df

display(dfs['aws'].head(5))
display(dfs['azure'].head(5))
display(dfs['gcp'].head(5))


### Step1: preprocessing data

# Keep only text in description column

# convert HTML to text
nltk.download('punkt')
def clean_html(text):
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
for operator in operator_lst:
    dfs[operator]['description'] = dfs[operator]['description'].apply(clean_html)
    # Lowercase
    dfs[operator]['description'] = dfs[operator]['description'].str.lower()
    # Remove extra spaces
    dfs[operator]['description'] = dfs[operator]['description'].str.replace(r'\s+', ' ', regex=True)
    # Remove leading and trailing spaces
    dfs[operator]['description'] = dfs[operator]['description'].str.strip()
    # Remove new line characters
    dfs[operator]['description'] = dfs[operator]['description'].str.replace('\n', ' ')
    # Remove tabs
    dfs[operator]['description'] = dfs[operator]['description'].str.replace('\t', ' ')
    # Remove '&nbsp;'
    dfs[operator]['description'] = dfs[operator]['description'].str.replace('&nbsp;', ' ')
     # Remove special characters
    dfs[operator]['description'] = dfs[operator]['description'].str.replace(r"[^a-z\s]", '', regex=True)
    
    # Tokenize
    dfs[operator]['description'] = dfs[operator]['description'].apply(word_tokenize)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    dfs[operator]['description'] = dfs[operator]['description'].apply(lambda x: [word for word in x if word not in stop_words])
    # Join tokens back to string
    dfs[operator]['description'] = dfs[operator]['description'].apply(lambda x: ' '.join(x))

    # Handling empty descriptions
    dfs[operator]['description'] = dfs[operator]['description'].replace('', 'unknown')

display(dfs['aws'].head(5))
display(dfs['azure'].head(5))
display(dfs['gcp'].head(5))

aws_reports = dfs['aws']['description'].tolist()
azure_reports = dfs['azure']['description'].tolist()
gcp_reports = dfs['gcp']['description'].tolist()

print('aws:', len(aws_reports))
print('azure:', len(azure_reports))
print('gcp:', len(gcp_reports))
print('total:', len(aws_reports) + len(azure_reports) + len(gcp_reports))


### Step2: Vectorize the text using TF-IDF

tfidf = TfidfVectorizer(max_features=1000)
X_aws = tfidf.fit_transform(aws_reports)
X_azure = tfidf.fit_transform(azure_reports)
X_gcp = tfidf.fit_transform(gcp_reports)


### Step3: K-means clustering

n_clusters = 5  # Adjust based on your dataset
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters_aws = kmeans.fit_predict(X_aws)
clusters_azure = kmeans.fit_predict(X_azure)
clusters_gcp = kmeans.fit_predict(X_gcp)


from sklearn.decomposition import PCA

cloud_data = {
    'AWS': (aws_reports, clusters_aws),
    'Azure': (azure_reports, clusters_azure),
    'GCP': (gcp_reports, clusters_gcp)
}

for provider, (reports, clusters) in cloud_data.items():
    df_cluster = pd.DataFrame({"description": reports, "cluster": clusters})
    print(f"{provider} Cluster Head:")
    print(df_cluster.head())

# Create 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Plot each provider in a subplot
for ax, (provider, (reports, clusters)) in zip(axes, cloud_data.items()):
    df_cluster = pd.DataFrame({"text": reports, "cluster": clusters})
    df_cluster['cluster'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title(f'{provider} Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Apply PCA
pca = PCA(n_components=2)
X_aws_pca = pca.fit_transform(X_aws.toarray())
X_azure_pca = pca.fit_transform(X_azure.toarray())
X_gcp_pca = pca.fit_transform(X_gcp.toarray())

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AWS plot
axes[0].scatter(X_aws_pca[:, 0], X_aws_pca[:, 1], c=clusters_aws, cmap="viridis")
axes[0].set_title("AWS K-means Clusters")
# Azure plot
axes[1].scatter(X_azure_pca[:, 0], X_azure_pca[:, 1], c=clusters_azure, cmap="viridis")
axes[1].set_title("Azure K-means Clusters")
# GCP plot
axes[2].scatter(X_gcp_pca[:, 0], X_gcp_pca[:, 1], c=clusters_gcp, cmap="viridis")
axes[2].set_title("GCP K-means Clusters")

# Show the combined plot
plt.tight_layout()
plt.show()


### Step4: Select sample data based on clusters

df_aws_cluster = pd.DataFrame({"description": aws_reports, "cluster": clusters_aws})
df_azure_cluster = pd.DataFrame({"description": azure_reports, "cluster": clusters_azure})
df_gcp_cluster = pd.DataFrame({"description": gcp_reports, "cluster": clusters_gcp})
# Save the clustered data
df_aws_cluster.to_csv('../data/3_sample_data/aws_clustered.csv', index=False)
df_azure_cluster.to_csv('../data/3_sample_data/azure_clustered.csv', index=False)
df_gcp_cluster.to_csv('../data/3_sample_data/gcp_clustered.csv', index=False)


# Sample the clusters

def sample_clusters(df, sample_fraction=0.2, scale=1.0):
    n_samples = int(scale * len(df) * sample_fraction)
    return df.groupby('cluster').apply(lambda x: x.sample(n=min(len(x), n_samples), random_state=42))
def get_index_lst(df):
    index_lst = df.index.tolist()
    index_lst = [i[1] for i in index_lst]
    return index_lst

df_aws_sample = sample_clusters(df_aws_cluster, sample_fraction=0.2, scale=0.2)
df_azure_sample = sample_clusters(df_azure_cluster, sample_fraction=0.2, scale=1.0)
df_gcp_sample = sample_clusters(df_gcp_cluster, sample_fraction=0.2, scale=0.1)

display(df_aws_sample)
aws_sample_index_lst = get_index_lst(df_aws_sample)
display(df_azure_sample)
azure_sample_index_lst = get_index_lst(df_azure_sample)
display(df_gcp_sample)
gcp_sample_index_lst = get_index_lst(df_gcp_sample)


print(len(aws_sample_index_lst), len(azure_sample_index_lst), len(gcp_sample_index_lst))

# select the sampled data from the cleaned data based on the index list
# Load clean data
operator_lst = ['aws', 'azure', 'gcp']
dfs = {}
for operator in operator_lst:
    df = pd.read_csv(f'../data/2_clean_data/{operator}.csv')
    dfs[operator] = df

aws_sample = dfs['aws'].iloc[aws_sample_index_lst]
azure_sample = dfs['azure'].iloc[azure_sample_index_lst]
gcp_sample = dfs['gcp'].iloc[gcp_sample_index_lst]
# save the sampled data
aws_sample.to_csv('../data/3_sample_data/aws_sample.csv', index=True)
azure_sample.to_csv('../data/3_sample_data/azure_sample.csv', index=True)
gcp_sample.to_csv('../data/3_sample_data/gcp_sample.csv', index=True)


### Step5: Clean HTML data of sample data

# Load sample data
operator_lst = ['aws', 'azure', 'gcp']
dfs = {}
for operator in operator_lst:
    df = pd.read_csv(f'../data/3_sample_data/{operator}_sample_html.csv', index_col=0)
    dfs[operator] = df

dfs['aws'].head(5)


def clean_html_text(text):
    """Convert HTML to plain text"""
    return BeautifulSoup(text, 'html.parser').get_text()

for operator in operator_lst:
    dfs[operator]['description'] = dfs[operator]['description'].apply(clean_html_text)

# remove space and \n of dfs['azure']['description']
dfs['azure']['description'] = dfs['azure']['description'].str.replace(r'\s+', ' ', regex=True)
dfs['azure']['description'] = dfs['azure']['description'].str.replace(r'\n', ' ', regex=True)


dfs['azure'].head(5)


# save the cleaned sample data
for operator in operator_lst:
    dfs[operator].to_csv(f'../data/3_sample_data/{operator}_sample.csv', index=True)