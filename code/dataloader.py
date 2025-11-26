# Update diseases pheonotype data loaders 
# update diectory to ablation study only
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import ast
import argparse
import os 

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def read_labels(label_col='diabetes_status', id_col='ID'):
    df = pd.read_csv('/projects/compbio/users/dyao8/2025/Embedding_prediction/newphenotypes/data/updated_disease_status_final(1020).csv')
    df = df.dropna(subset=[label_col, id_col])
    df['label'] = df[label_col].astype(int)
    return df[[id_col, 'label']]

def read_embeddings(embedding_csv, embedding_col, id_col='ID'):
    df = pd.read_csv(embedding_csv, usecols=[id_col, embedding_col])
    df[embedding_col] = df[embedding_col].apply(ast.literal_eval)
    print(df.head())
    return df

def merge_label_embedding(embeddings, labels, id_col='ID'):
    merged = pd.merge(embeddings, labels, on=id_col, how='inner')
    print(f"Merged shape: {merged.shape}")
    return merged

def split_by_ids(df, id_col, train_id_file, test_id_file):
    train_ids = pd.read_csv(train_id_file, usecols=['IID'], sep='\t')['IID'].values
    test_ids = pd.read_csv(test_id_file, usecols=['IID'], sep='\t')['IID'].values
    print(train_ids[:5])
    train = df[df[id_col].isin(train_ids)]
    test = df[df[id_col].isin(test_ids)]
    val = df[~df[id_col].isin(np.concatenate([train_ids, test_ids]))]
    print(f"Train/Val/Test split shapes: {train.shape}, {val.shape}, {test.shape}")
    return train, val, test


def save_tensors(path, X, y, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'X': X, 'y': y, 'IDs': ids}, path)
    print(f"Saved tensors")
    
def to_tensor(arr):
    return torch.tensor(np.stack(arr), dtype=torch.float32)

def pipeline(
    disease,
    embedding_csv,
    embedding_col,

    model_name,
    output_dir,
    id_col='ID'
):
    device = get_device()
    #output_dir=os.path.join(f'/projects/compbio/users/dyao8/2025/Ablation_study/output/top_random')
    label_col =f'{disease}_status'
    labels = read_labels(label_col, id_col)
    embeddings = read_embeddings(embedding_csv, embedding_col, id_col)
    merged = merge_label_embedding(embeddings, labels, id_col)
    disease_cap = disease.capitalize()
    train_id_file = f'/projects/compbio/users/dyao8/2025/Embedding_prediction/all_ukb_mutl_lable/T2d/Dataloader/train_ids_cleaned.txt'
    test_id_file = f'/projects/compbio/users/dyao8/2025/Embedding_prediction/all_ukb_mutl_lable/T2d/Dataloader/test_ids_cleaned.txt'


    train, val, test = split_by_ids(merged, id_col, train_id_file, test_id_file)
    save_dir = os.path.join(output_dir, disease)
    os.makedirs(save_dir, exist_ok=True)

    # Convert to tensors
    for split, df in zip(['train', 'val', 'test'], [train, val, test]):
        X_tensor = to_tensor(df[embedding_col].values)
        y_tensor = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)
        id_array = df[id_col].values
        split_dir = os.path.join(save_dir, f'{disease}_{split}.pt')
        save_tensors(split_dir, X_tensor, y_tensor, id_array)

# Example usage:
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run data processing pipeline")
    parser.add_argument('--disease', type=str, required=True, help='Disease name (e.g., diabetes)')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., Gemini)')
    parser.add_argument('--embeddingcol', type=str, required=True, help='Path to the embedding column in the CSV file')
    parser.add_argument('--embeddingfile', type=str, required=True, help='embeddingfile path')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')

    args = parser.parse_args()   
    pipeline(
        disease=args.disease,
        model_name=args.model,
        output_dir=args.output_dir,
        embedding_csv=args.embeddingfile,
        embedding_col=args.embeddingcol,
        id_col='ID',
    )