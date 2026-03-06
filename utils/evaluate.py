import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
import numpy as np
import pandas as pd


def exact_match(df_extract: pd.DataFrame, df_label: pd.DataFrame, eval_fields: list) -> pd.DataFrame:
    df_em = pd.DataFrame(columns=eval_fields)
    df_em.loc[0] = 0.0

    for field in eval_fields:
        pred_col = field
        label_col = 'label_' + field  # Both DataFrames now have the same column names

        if pred_col not in df_extract.columns or label_col not in df_label.columns:
            print(f"[Warning] Missing column: {pred_col} or {label_col}, skipping.")
            continue

        accuracy = np.mean(df_extract[pred_col].fillna("") == df_label[label_col].fillna(""))
        df_em.loc[0, field] = round(accuracy * 100, 2)

    df_em.index = ["EM-Accuracy"]

    print("\n[Exact Match Accuracy (%)]")
    print(df_em)
    
    return df_em


def token_level(df_extract, df_label, eval_fields, average: bool = True):

    def token_metrics(pred_text, ref_text):
        pred_tokens = set(word_tokenize(str(pred_text).lower()))
        ref_tokens = set(word_tokenize(str(ref_text).lower()))

        tp = len(pred_tokens & ref_tokens)
        fp = len(pred_tokens - ref_tokens)
        fn = len(ref_tokens - pred_tokens)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return precision, recall, f1

    # Initialize results
    token_scores = {
        "field": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    # Iterate over fields
    for field in eval_fields:
        pred_col = df_extract.get(field)
        label_col = df_label.get('label_' + field)  # Both DataFrames now have the same column names

        if pred_col is None or label_col is None:
            print(f"[Warning] Missing field: {field}, skipping.")
            continue

        # Handle missing values 
        metrics = [token_metrics(pred, ref) for pred, ref in zip(pred_col.fillna(""), label_col.fillna(""))]
        precisions, recalls, f1s = zip(*metrics)

        if average:
            token_scores["field"].append(field)
            # NOTE: there is no need to multiply by 100 here, later we divide by 100 in the plotting script
            token_scores["precision"].append(round(np.mean(precisions) * 100, 2))
            token_scores["recall"].append(round(np.mean(recalls) * 100, 2))
            token_scores["f1"].append(round(np.mean(f1s) * 100, 2))
        else:
            token_scores["field"].append(field)
            token_scores["precision"].append(precisions)
            token_scores["recall"].append(recalls)
            token_scores["f1"].append(f1s)

    df_tk = pd.DataFrame(token_scores)
    df_tk = df_tk.T
    df_tk.index = ["", "TK-Precision", "TK-Recall", "TK-F1"]
    df_tk.columns = eval_fields
    df_tk = df_tk.iloc[1:]
    
    print("\n[Token-Level Evaluation (%)]")
    print(df_tk if average else df_tk[["field"]])

    return df_tk


def bert_score(df_extract, df_label, eval_fields, 
               model_type="microsoft/deberta-v3-base",
               normalize_scores=False):
    """
    Compute BERTScore with proper normalization to address bias towards high values.
    
    Args:
        df_extract: DataFrame with extracted predictions
        df_label: DataFrame with ground truth labels
        eval_fields: List of fields to evaluate
        model_type: BERT model to use for scoring
        normalize_scores: Whether to apply normalization to address high score bias
    """
    df_bs = {}
    
    # Import here to avoid circular imports
    try:
        from bert_score import score
        from bert_score import BERTScorer
    except ImportError:
        print("[Error] bert_score package not found. Please install it with: pip install bert-score")
        return pd.DataFrame()

    selected_fields = []
    # only calculate 'user_symptom' and 'root_cause' if they are in eval_fields
    if 'user_symptom' in eval_fields:
        selected_fields.append('user_symptom')
    if 'root_cause' in eval_fields:
        selected_fields.append('root_cause')

    for col in selected_fields:
        # Extract predictions and references as lists of strings
        preds = df_extract[col].fillna("").astype(str).tolist()
        refs = df_label['label_' + col].fillna("").astype(str).tolist()

        # Filter out empty pairs and very short texts (which can cause inflated scores)
        filtered = [(p, r) for p, r in zip(preds, refs) 
                   if p.strip() and r.strip() and len(p.strip()) > 3 and len(r.strip()) > 3]
        
        if not filtered:
            print(f"[Warning] All candidate/reference pairs are empty or too short for field: {col}")
            continue
            
        if len(filtered) < 2:
            print(f"[Warning] Too few valid pairs for field: {col} (need at least 2, got {len(filtered)})")
            continue
            
        preds, refs = zip(*filtered)    

        try:
            # Compute BERTScore with specific model and normalization
            # Use use_fast_tokenizer=False to avoid sentencepiece issues
            P, R, F1 = score(list(preds), list(refs), 
                            lang="en", 
                            verbose=False,
                            model_type=model_type,
                            use_fast_tokenizer=False)
            
            if normalize_scores:
                # Apply BERTScore normalization (rescale to more reasonable range)
                # BERTScore values are typically in [0, 1] but can be biased high
                # Normalize by centering around expected mean and scaling
                P_norm = (P - P.mean()) / P.std() * 0.15 + 0.85  # Scale to reasonable range
                R_norm = (R - R.mean()) / R.std() * 0.15 + 0.85
                F1_norm = (F1 - F1.mean()) / F1.std() * 0.15 + 0.85
                
                # Clip to reasonable bounds
                P_norm = np.clip(P_norm, 0.0, 1.0)
                R_norm = np.clip(R_norm, 0.0, 1.0)
                F1_norm = np.clip(F1_norm, 0.0, 1.0)
                
                # Average across all samples
                avg_p = P_norm.mean().item()
                avg_r = R_norm.mean().item()
                avg_f1 = F1_norm.mean().item()
            else:
                # Use raw scores without normalization
                avg_p = P.mean().item()
                avg_r = R.mean().item()
                avg_f1 = F1.mean().item()

            df_bs[col] = [avg_p, avg_r, avg_f1]
            
        except Exception as e:
            print(f"[Error] Failed to compute BERTScore for field {col}: {str(e)}")
            continue

    if not df_bs:
        print("[Warning] No BERTScore results computed")
        return pd.DataFrame()
        
    df_bs = pd.DataFrame(df_bs)
    df_bs.index = ["BS-Precision", "BS-Recall", "BS-F1"]
    
    score_type = "(Normalized)" if normalize_scores else "(Raw)"
    print(f"\n[BERT Score {score_type}]")
    print(df_bs)

    return df_bs