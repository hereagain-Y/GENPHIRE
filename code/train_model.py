#!/usr/bin/env python3
"""
GENPHIRE - Complete Training Pipeline
Uses EXACT model architecture and loss from original model_tranning.py
Simplified for easy testing with any embedding data.

Usage:
    python train_model.py \
        --input data/toy_embeddings.csv \
        --output_dir results \
        --phenotype simulated_disease
        
Author: Yao Lab, Emory University (2025)
"""

import os
import ast
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    precision_score, recall_score, brier_score_loss,
    average_precision_score, matthews_corrcoef,
    precision_recall_curve, classification_report
)


# ==================== EXACT Model from model_tranning.py ====================

class FlexibleMLP(nn.Module):
    """EXACT model architecture from model_tranning.py"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max(1, hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_dim // 4), output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== EXACT Loss from model_tranning.py ====================

class DynamicWeightedLoss(nn.Module):
    """EXACT loss function from model_tranning.py"""
    def __init__(
        self,
        loss_type: str = "weighted_bce",
        f_case: Optional[torch.Tensor] = None,
        f_control: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.device = device

        if f_case is None or f_control is None:
            self.f_case = None
            self.f_control = None
        else:
            f_case = torch.as_tensor(f_case, dtype=torch.float32, device=device)
            f_control = torch.as_tensor(f_control, dtype=torch.float32, device=device)
            self.f_case = torch.clamp(f_case, min=1e-6)
            self.f_control = torch.clamp(f_control, min=1e-6)

        if loss_type == "weighted_bce":
            if self.f_case is not None and self.f_control is not None:
                self.w_case = self.f_control / self.f_case
            else:
                self.w_case = torch.tensor(1.0, dtype=torch.float32, device=device)
            self.w_control = torch.tensor(1.0, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy(probs, targets, reduction="none")
        
        if isinstance(self.w_case, torch.Tensor) and self.w_case.dim() > 0:
            weight_pos = self.w_case
            weight_neg = self.w_control
        else:
            weight_pos = self.w_case if self.w_case is not None else torch.tensor(1.0, device=targets.device)
            weight_neg = self.w_control if self.w_control is not None else torch.tensor(1.0, device=targets.device)
        weight = torch.where(targets > 0, weight_pos, weight_neg)
        
        return (bce * weight).mean()


def compute_case_control_counts(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """EXACT function from model_tranning.py"""
    if y.ndim == 1:
        y_vec = y
    elif y.ndim == 2 and y.shape[1] == 1:
        y_vec = y.squeeze(1)
    else:
        pos = y.sum(dim=0)
        neg = y.shape[0] - pos
        return pos, neg
    pos = y_vec.sum()
    neg = y_vec.numel() - pos
    return pos, neg


# ==================== Device Setup ====================

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ==================== Data Loading ====================

def load_embeddings(csv_file, embedding_col='embedding', id_col='ID'):
    """Load embeddings from CSV file."""
    print(f"\nLoading embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert string representation to list
    df[embedding_col] = df[embedding_col].apply(ast.literal_eval)
    
    # Extract embeddings as numpy array
    embeddings = np.stack(df[embedding_col].values)
    ids = df[id_col].values
    
    print(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
    return embeddings, ids, df


def simulate_phenotype_labels(n_samples, n_classes=2, seed=42):
    """
    Simulate phenotype labels for demonstration.
    In real use, replace with actual phenotype data.
    """
    np.random.seed(seed)
    
    if n_classes == 2:
        # Binary classification (e.g., disease vs healthy)
        # Simulate ~20% positive rate (realistic for diseases)
        labels = np.random.binomial(1, 0.2, n_samples)
        print(f"\nSimulated binary labels: {labels.sum()} positive, {len(labels)-labels.sum()} negative")
    else:
        # Multi-class classification
        labels = np.random.randint(0, n_classes, n_samples)
        print(f"\nSimulated {n_classes}-class labels")
    
    return labels


def split_and_save_data(X, y, ids, output_dir, phenotype_name, test_size=0.2, val_size=0.1, seed=42):
    """
    Split data into train/val/test and save as .pt files.
    Format matches dataloader.py expectations.
    """
    print(f"\nSplitting data (test={test_size}, val={val_size})...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} positive)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} positive)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} positive)")
    
    # Create output directory matching dataloader.py structure
    save_dir = Path(output_dir) / phenotype_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to tensors and save (matching dataloader.py format)
    splits = {
        'train': (X_train, y_train, ids_train),
        'val': (X_val, y_val, ids_val),
        'test': (X_test, y_test, ids_test)
    }
    
    for split_name, (X_split, y_split, ids_split) in splits.items():
        X_tensor = torch.tensor(X_split, dtype=torch.float32)
        y_tensor = torch.tensor(y_split, dtype=torch.float32).unsqueeze(1)
        
        save_path = save_dir / f"{phenotype_name}_{split_name}.pt"
        torch.save({
            'X': X_tensor,
            'y': y_tensor,
            'IDs': ids_split
        }, save_path)
        print(f"  Saved: {save_path}")
    
    return save_dir


def load_pt_data(file_path, device):
    """Load .pt file matching dataloader.py format."""
    data = torch.load(file_path, map_location=device, weights_only=False)
    return data['X'], data['y'], data.get('IDs')


# ==================== Training (matching model_tranning.py) ====================

def train_model(
    model, train_loader, val_loader,
    loss_fn, optimizer,
    num_epochs, patience, device
):
    """Train model with early stopping - matches model_tranning.py logic."""
    print(f"\nTraining for up to {num_epochs} epochs (patience={patience})...")
    
    best_val_loss = float('inf')
    no_improve = 0
    best_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item()
        
        val_loss /= max(1, len(val_loader))
        
        # Print progress
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        
        # Early stopping check (matching model_tranning.py)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"  Best validation loss: {best_val_loss:.4f}")
    return model


# ==================== Evaluation ====================

def _optimal_threshold_for_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Find optimal threshold that maximizes F1 score (from model_tranning.py)."""
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = 2 * (p * r) / (p + r + 1e-12)
    return (t[np.nanargmax(f1)] if t.size > 0 else 0.5)


def evaluate_model(model, val_loader, test_loader, device):
    """
    Evaluate model using validation set to find threshold,
    then apply to test set (matching model_tranning.py approach).
    """
    print("\nDetermining optimal threshold from validation set...")
    
    # Get validation predictions
    model.eval()
    val_logits_list = []
    val_true_list = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            val_logits_list.append(logits.cpu().numpy())
            val_true_list.append(yb.cpu().numpy())
    
    val_logits = np.concatenate(val_logits_list, axis=0).flatten()
    val_true = np.concatenate(val_true_list, axis=0).flatten().astype(int)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    
    # Find optimal threshold (matching model_tranning.py)
    optimal_threshold = _optimal_threshold_for_f1(val_true, val_probs)
    print(f"  Optimal threshold (F1-based): {optimal_threshold:.4f}")
    
    # Get test predictions
    print("Evaluating on test set...")
    test_logits_list = []
    test_true_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            test_logits_list.append(logits.cpu().numpy())
            test_true_list.append(yb.cpu().numpy())
    
    test_logits = np.concatenate(test_logits_list, axis=0).flatten()
    test_true = np.concatenate(test_true_list, axis=0).flatten().astype(int)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    
    # Apply threshold
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics (matching model_tranning.py)
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(test_true, test_preds),
        'roc_auc': roc_auc_score(test_true, test_probs),
        'pr_auc': average_precision_score(test_true, test_probs),
        'brier': brier_score_loss(test_true, test_probs),
        'f1_score': f1_score(test_true, test_preds, zero_division=0),
        'precision': precision_score(test_true, test_preds, zero_division=0),
        'recall': recall_score(test_true, test_preds, zero_division=0),
        'mcc': matthews_corrcoef(test_true, test_preds),
        'n_samples': len(test_true),
        'n_positive': int(test_true.sum()),
        'n_negative': int(len(test_true) - test_true.sum()),
    }
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"  Threshold: {metrics['threshold']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  MCC:       {metrics['mcc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print("="*60)
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=['Control', 'Case']))
    
    return metrics, test_probs, test_preds, test_true


# ==================== Data Loading ====================

def load_embeddings(csv_file, embedding_col='embedding', id_col='ID'):
    """Load embeddings from CSV file."""
    print(f"\nLoading embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert string representation to list
    if df[embedding_col].dtype == 'object':
        df[embedding_col] = df[embedding_col].apply(ast.literal_eval)
    
    # Extract embeddings as numpy array
    embeddings = np.stack(df[embedding_col].values)
    ids = df[id_col].values
    
    print(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]} dimensions")
    return embeddings, ids, df


def simulate_phenotype_labels(n_samples, n_classes=2, seed=42):
    """Simulate phenotype labels for demonstration."""
    np.random.seed(seed)
    
    if n_classes == 2:
        labels = np.random.binomial(1, 0.2, n_samples)
        print(f"\nSimulated binary labels: {labels.sum()} positive, {len(labels)-labels.sum()} negative")
    else:
        labels = np.random.randint(0, n_classes, n_samples)
        print(f"\nSimulated {n_classes}-class labels")
    
    return labels


def split_and_save_data(X, y, ids, output_dir, phenotype_name, test_size=0.2, val_size=0.1, seed=42):
    """Split data and save as .pt files (matching dataloader.py format)."""
    print(f"\nSplitting data (test={test_size}, val={val_size})...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} positive, {len(y_train)-y_train.sum()} negative)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} positive, {len(y_val)-y_val.sum()} negative)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} positive, {len(y_test)-y_test.sum()} negative)")
    
    # Create output directory (matching dataloader.py structure)
    save_dir = Path(output_dir) / phenotype_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to tensors and save (EXACT format from dataloader.py)
    splits = {
        'train': (X_train, y_train, ids_train),
        'val': (X_val, y_val, ids_val),
        'test': (X_test, y_test, ids_test)
    }
    
    for split_name, (X_split, y_split, ids_split) in splits.items():
        X_tensor = torch.tensor(X_split, dtype=torch.float32)
        y_tensor = torch.tensor(y_split, dtype=torch.float32).unsqueeze(1)
        
        save_path = save_dir / f"{phenotype_name}_{split_name}.pt"
        torch.save({
            'X': X_tensor,
            'y': y_tensor,
            'IDs': ids_split
        }, save_path)
        print(f"    Saved: {save_path.name}")
    
    return save_dir


def load_pt_data(file_path, device):
    """Load .pt file (matching dataloader.py format)."""
    data = torch.load(file_path, map_location=device, weights_only=False)
    return data['X'], data['y'], data.get('IDs')


def save_results(metrics, probs, preds, labels, ids, output_dir, phenotype_name):
    """Save predictions and metrics."""
    results_dir = Path(output_dir) / phenotype_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions (matching model_tranning.py format)
    pred_df = pd.DataFrame({
        'participant_id': ids,
        'true_label': labels.astype(int),
        'predicted_probability': probs,
        'predicted_label': preds.astype(int),
        'phenotype': phenotype_name
    })
    pred_file = results_dir / f"{phenotype_name}_predictions_{timestamp}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"\nSaved predictions: {pred_file.name}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_file = results_dir / f"{phenotype_name}_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics: {metrics_file.name}")
    
    return pred_file, metrics_file


# ==================== Main Pipeline ====================

def main():
    parser = argparse.ArgumentParser(
        description="GENPHIRE Training Pipeline - Uses exact model/loss from model_tranning.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python train_model.py \
      --input data/toy_embeddings.csv \
      --labels data/toy_phenotype.csv \
      --output_dir results \
      --phenotype diabetes
        """
    )
    
    # Data arguments
    parser.add_argument('--input', required=True, help='Path to embeddings CSV file')
    parser.add_argument('--labels', required=True, help='Path to phenotype labels CSV file')
    parser.add_argument('--label_col', default='disease_status', help='Label column name in labels file')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--phenotype', default='simulated_disease', help='Phenotype/disease name')
    parser.add_argument('--embedding_col', default='embedding', help='Embedding column name')
    parser.add_argument('--id_col', default='ID', help='ID column name')
    
    # Split arguments
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments (matching model_tranning.py defaults)
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = get_device()
    
    print("\n" + "="*80)
    print("GENPHIRE - Disease Prediction Training Pipeline")
    print("="*80)
    print(f"Input:       {args.input}")
    print(f"Phenotype:   {args.phenotype}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Model:       FlexibleMLP (hidden={args.hidden_dim}, dropout={args.dropout})")
    print(f"Loss:        DynamicWeightedLoss (weighted_bce)")
    print("="*80)
    
    # Step 1: Load embeddings
    X, ids, df = load_embeddings(args.input, embedding_col=args.embedding_col, id_col=args.id_col)
    
    # Step 2: Load phenotype labels
    print(f"\nLoading phenotype labels from: {args.labels}")
    labels_df = pd.read_csv(args.labels)
    print(f"Loaded {len(labels_df)} labels")
    
    # Merge with embeddings on ID
    merged_df = df[[args.id_col, args.embedding_col]].merge(
        labels_df[[args.id_col, args.label_col]], 
        on=args.id_col, 
        how='inner'
    )
    print(f"Matched {len(merged_df)} samples with both embeddings and labels")
    
    # Extract matched data
    X = np.stack(merged_df[args.embedding_col].values)
    y = merged_df[args.label_col].values
    ids = merged_df[args.id_col].values
    
    print(f"  Cases (positive): {y.sum()}")
    print(f"  Controls (negative): {len(y) - y.sum()}")
    
    # Step 3: Split and save data
    data_dir = split_and_save_data(
        X, y, ids, args.output_dir, args.phenotype,
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    
    # Step 4: Load data (matching dataloader.py approach)
    train_path = data_dir / f"{args.phenotype}_train.pt"
    val_path = data_dir / f"{args.phenotype}_val.pt"
    test_path = data_dir / f"{args.phenotype}_test.pt"
    
    X_train, y_train, ids_train = load_pt_data(train_path, device)
    X_val, y_val, ids_val = load_pt_data(val_path, device)
    X_test, y_test, ids_test = load_pt_data(test_path, device)
    
    # Step 5: Create dataloaders (matching model_tranning.py)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=args.batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Step 6: Initialize model (EXACT FlexibleMLP from model_tranning.py)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if y_train.ndim == 2 else 1
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    model = FlexibleMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nModel Architecture (FlexibleMLP):")
    print(f"  Input:  {input_dim}")
    print(f"  Hidden: {args.hidden_dim} → {args.hidden_dim//2} → {max(1, args.hidden_dim//4)}")
    print(f"  Output: {output_dim}")
    print(f"  Dropout: {args.dropout}")
    
    # Step 7: Setup optimizer and loss (EXACT from model_tranning.py)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Compute case/control counts for weighted loss (matching model_tranning.py)
    pos_counts, neg_counts = compute_case_control_counts(y_train)
    print(f"\nTraining set class distribution:")
    print(f"  Cases (positive):    {pos_counts.item():.0f}")
    print(f"  Controls (negative): {neg_counts.item():.0f}")
    print(f"  Case weight: {(neg_counts/pos_counts).item():.2f}")
    
    loss_fn = DynamicWeightedLoss(
        loss_type="weighted_bce",
        f_case=pos_counts,
        f_control=neg_counts,
        device=device,
    )
    
    # Step 8: Train model
    model = train_model(
        model, train_loader, val_loader,
        loss_fn, optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device
    )
    
    # Step 9: Evaluate model
    metrics, probs, preds, labels = evaluate_model(model, val_loader, test_loader, device)
    
    # Step 10: Save results
    ids_test_np = ids_test if not torch.is_tensor(ids_test) else ids_test.cpu().numpy()
    pred_file, metrics_file = save_results(
        metrics, probs, preds, labels,
        ids_test_np, args.output_dir, args.phenotype
    )
    
    # Step 11: Save model (optional)
    if args.save_model:
        model_file = Path(args.output_dir) / args.phenotype / f"best_model_weighted_bce.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Saved model: {model_file.name}")
    
    print("\n" + "="*80)
    print("Pipeline Complete! ✓")
    print("="*80)
    print(f"\nResults saved in: {args.output_dir}/{args.phenotype}/")
    print(f"  - Data splits:  {args.phenotype}_{{train,val,test}}.pt")
    print(f"  - Predictions:  {pred_file.name}")
    print(f"  - Metrics:      {metrics_file.name}")
    if args.save_model:
        print(f"  - Model:        best_model_weighted_bce.pth")
    print()


if __name__ == '__main__':
    main()
