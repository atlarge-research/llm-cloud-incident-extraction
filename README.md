# LLM for Cloud Incident Data Extraction

## Setup
Setup a Python environment and install requirements to run the scripts for (re-)producing results.

```
conda create -n llm-for-report-extraction
conda activate llm-for-report-extraction
pip install -r requirements.txt
```
## Experiment
Run extraction and evaluation
```
python main.py  # select the function(extraction/evaluation) in main.py
```
*Note: For Section 4.1, the extraction results are produced on November 13, 2025. For Sections 4.2, 4.3, and 4.4, the extraction results are produced on on August 27, 2025.*

Produce figures and tables
```
cd plotting
python figure_acc_bs.py  # select which plot to (re-)produce
```


## File Tree
```
llm-for-report-extraction/
│
├── data/                     # Datasets (AWS, AZURE, GCP)
│   ├── 1_raw_data/           # Original incident reports
│   ├── 2_clean_data/         # Processed clean data
│   ├── 3_sample_data/        # Sampled data by K-means clustering
│   ├── 4_label_data/         # Annotated data for evaluation
│   └── data_process.py       # Data process, clean, and sample
│
├── evaluation/               # Evaluate pipeline
│   ├── run_evaluation_for_prompt.py   
│   └── run_evaluation.py        
│
├── extraction/               # Extract pipeline
│   ├── run_extraction_for_analysis.py  
│   ├── run_extraction_for_prompt.py        
│   └── run_extraction.py        
│
├── models/                   # LLM API wrappers
│   ├── gpt_api.py
│   ├── claude_api.py
│   └── gemini_api.py
│
├── plotting/                 # Plotting all figures and tables in the paper
│
├── prompts/                  # Prompt templates, instructions, and strategieas
│
├── results/                  # Output files
│   ├── analysis/             # Extracted files jsonl for analysis
│   ├── evaluations/          # Accuracy evaluation results csv
│   ├── extractions/          # Extracted files jsonl, and model performance csv
│   ├── figures/              # Figures for paper
│   ├── prompts/              # Results for prompt strategies 
│   └── tables/               # Tables for paper
│
├── utils/                    # Utility functions
│   ├── io.py                 # Read/write dataframes
│   ├── extract.py            # API Calling functions
│   ├── evaluate.py           # Evaluation methods
│   └── plotting.py           # Plotting functions
│
├── config.yaml               # API keys and models setting
├── main.py                   # Entry point for running the full pipeline
├── README.md                 # Project overview
└── requirements.txt          # Project requirements
```
