"""
# Oct 21, 2025
Unified Disease Prediction Training & Evaluation Pipeline
- Bootstrap CIs with fixed threshold from validation
- Individual participant predictions saved
- Comprehensive metrics including Brier score
- Clean output organization

Usage:
    python unified_pipeline.py --data_root ./data --diseases afib,diabetes,cvd \
        --output_dir ./results --save_model --n_boot 1000
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader, TensorDataset


# ==================== Data Loading ====================

def load_pt_dataset(file_path: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    data = torch.load(file_path, map_location=device, weights_only=False)
    X = data["X"]
    y = data["y"]
    ids = data.get("IDs")
    return X, y, ids


def create_dataloaders(
    data_root: str,
    disease: str,
    batch_size: int,
    device: str,
) -> Tuple[DataLoader, DataLoader, DataLoader,
           torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    train_path = os.path.join(data_root, disease, f"{disease}_train.pt")
    val_path   = os.path.join(data_root, disease, f"{disease}_val.pt")
    test_path  = os.path.join(data_root, disease, f"{disease}_test.pt")

    X_train, y_train, train_ids = load_pt_dataset(train_path, device=device)
    X_val,   y_val,   val_ids   = load_pt_dataset(val_path, device=device)
    X_test,  y_test,  test_ids  = load_pt_dataset(test_path, device=device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader,
            X_train, y_train, train_ids, val_ids, test_ids)


# ==================== Model Architecture ====================

class FlexibleMLP(nn.Module):
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


# ==================== Loss Function ====================

class DynamicWeightedLoss(nn.Module):
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


# ==================== Threshold & Metrics ====================

def _optimal_threshold_for_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Find optimal threshold that maximizes F1 score."""
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = 2 * (p * r) / (p + r + 1e-12)
    return (t[np.nanargmax(f1)] if t.size > 0 else 0.5)


def _compute_all_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute all metrics at a fixed threshold."""
    y_pred = (probs >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "brier": brier_score_loss(y_true, probs),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def bootstrap_metric_cis(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    stratified: bool = True,
    random_state: int = 42,
    fixed_threshold: float = 0.5,
    ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Bootstrap CIs for all metrics using a FIXED threshold.
    Returns: (point_estimates, ci_dict)
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs).astype(float)

    # Point estimate
    point = _compute_all_metrics(y, p, fixed_threshold)

    # Build index groups for resampling
    if ids is not None:
        ids = np.asarray(ids)
        if stratified:
            pos_ids = np.unique(ids[y == 1])
            neg_ids = np.unique(ids[y == 0])
        else:
            pos_ids = np.unique(ids)
            neg_ids = np.array([], dtype=pos_ids.dtype)

    buckets = defaultdict(list)
    q_low = (1 - ci) / 2
    q_high = 1 - q_low

    for _ in range(n_boot):
        if ids is not None:
            # Cluster bootstrap
            if stratified:
                b_pos_ids = rng.choice(pos_ids, size=len(pos_ids), replace=True)
                b_neg_ids = rng.choice(neg_ids, size=len(neg_ids), replace=True)
                b_ids = np.concatenate([b_pos_ids, b_neg_ids])
            else:
                b_ids = rng.choice(np.unique(ids), size=len(np.unique(ids)), replace=True)
            mask = np.isin(ids, b_ids)
            y_b, p_b = y[mask], p[mask]
        else:
            # Row-level bootstrap
            if stratified:
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
                b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
                idx = np.concatenate([b_pos, b_neg])
            else:
                idx = rng.choice(np.arange(len(y)), size=len(y), replace=True)
            y_b, p_b = y[idx], p[idx]

        # Compute metrics with FIXED threshold
        m = _compute_all_metrics(y_b, p_b, fixed_threshold)
        for k, v in m.items():
            buckets[k].append(v)

    ci_dict = {k: (float(np.quantile(v, q_low)), float(np.quantile(v, q_high))) 
               for k, v in buckets.items()}
    return point, ci_dict


# ==================== Save Outputs ====================

def save_participant_predictions(
    ids: Optional[np.ndarray],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease: str,
    loss_name: str,
    output_dir: str,
) -> str:
    """Save individual participant predictions."""
    if ids is None:
        ids = np.arange(len(y_true))
    else:
        ids = np.asarray(ids)
    
    y_true = y_true.flatten() if y_true.ndim > 1 else y_true
    y_prob = y_prob.flatten() if y_prob.ndim > 1 else y_prob
    
    df = pd.DataFrame({
        'participant_id': ids,
        'true_label': y_true.astype(int),
        'predicted_probability': y_prob,
        'disease': disease,
        'loss_type': loss_name,
    })
    
    # Save to disease-specific directory
    disease_dir = os.path.join(output_dir, disease)
    os.makedirs(disease_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{disease}_{loss_name}_predictions_{timestamp}.csv"
    filepath = os.path.join(disease_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"  Saved predictions: {filename}")
    return filepath


def save_roc_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease: str,
    loss_name: str,
    output_dir: str,
) -> str:
    """Save ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'model_name': 'MLP',
        'disease': disease,
        'loss_name': loss_name,
        'auc_score': roc_auc,
    })
    
    disease_dir = os.path.join(output_dir, disease)
    os.makedirs(disease_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{disease}_{loss_name}_roc_curve_{timestamp}.csv"
    filepath = os.path.join(disease_dir, filename)
    df.to_csv(filepath, index=False)
    
    return filepath


# ==================== Training Function ====================

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    patience: int,
    device: str,
    disease: str,
    loss_name: str,
    output_dir: str,
    n_boot: int,
    ci: float,
    stratified_boot: bool,
    test_ids: Optional[np.ndarray],
    val_ids: Optional[np.ndarray],
) -> Tuple[Dict[str, float], float]:
    """
    Train model and evaluate with bootstrap CIs.
    Returns: (metrics_dict, optimal_threshold)
    """
    print(f"\n{'='*60}")
    print(f"Training: {disease} - {loss_name}")
    print(f"{'='*60}")
    
    best_val = float("inf")
    no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))

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

        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if "best_state" in locals():
        model.load_state_dict(best_state)

    # ==================== STEP 1: Get validation predictions & determine threshold ====================
    print("\n  Computing optimal threshold from validation set...")
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
    
    # Find optimal threshold
    optimal_threshold = _optimal_threshold_for_f1(val_true, val_probs)
    print(f"  Optimal threshold (F1-based): {optimal_threshold:.4f}")
    
    # ==================== STEP 2: Get test predictions ====================
    print("  Evaluating on test set...")
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
    
    # ==================== STEP 3: Save individual predictions ====================
    test_ids_np = None if test_ids is None else (
        test_ids.cpu().numpy() if torch.is_tensor(test_ids) else np.asarray(test_ids)
    )
    
    pred_file = save_participant_predictions(
        test_ids_np, test_true, test_probs, disease, loss_name, output_dir
    )
    
    # ==================== STEP 4: Compute metrics with bootstrap CIs ====================
    print(f"  Computing metrics with {n_boot} bootstrap samples...")
    point, cis = bootstrap_metric_cis(
        test_true, test_probs,
        n_boot=n_boot, ci=ci, stratified=stratified_boot,
        random_state=123, fixed_threshold=optimal_threshold,
        ids=test_ids_np,
    )
    
    # ==================== STEP 5: Save ROC data ====================
    save_roc_data(test_true, test_probs, disease, loss_name, output_dir)
    
    # ==================== STEP 6: Format results ====================
    metrics = {
        "threshold": optimal_threshold,
        "n_samples": len(test_true),
        "n_positive": int(test_true.sum()),
        "n_negative": int(len(test_true) - test_true.sum()),
        "prediction_file": os.path.basename(pred_file),
    }
    
    for k, v in point.items():
        metrics[k] = float(v)
        if k in cis:
            lo, hi = cis[k]
            metrics[f"{k}_ci_lower"] = float(lo)
            metrics[f"{k}_ci_upper"] = float(hi)
            metrics[f"{k}_ci_formatted"] = f"{v:.3f} ({lo:.3f}-{hi:.3f})"
    
    # Print summary
    print(f"\n  Results Summary:")
    print(f"    ROC-AUC:  {metrics.get('roc_auc_ci_formatted', 'N/A')}")
    print(f"    PR-AUC:   {metrics.get('pr_auc_ci_formatted', 'N/A')}")
    print(f"    Brier:    {metrics.get('brier_ci_formatted', 'N/A')}")
    print(f"    MCC:      {metrics.get('mcc_ci_formatted', 'N/A')}")
    print(f"    F1:       {metrics.get('f1_score_ci_formatted', 'N/A')}")
    
    return metrics, optimal_threshold


# ==================== Main Pipeline ====================

@dataclass
class TrainConfig:
    data_root: str
    diseases: List[str]
    output_dir: str
    device: str
    batch_size: int = 512
    hidden_dim: int = 512
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 500
    patience: int = 30
    save_model: bool = False
    loss_type: str = "weighted_bce"
    n_boot: int = 1000
    ci: float = 0.95
    stratified_boot: bool = True


def run_pipeline(cfg: TrainConfig) -> pd.DataFrame:
    """Main training pipeline."""
    device = cfg.device
    results_rows = []
    
    print(f"\n{'='*80}")
    print("Disease Prediction Training Pipeline")
    print(f"{'='*80}")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Diseases: {', '.join(cfg.diseases)}")
    print(f"Loss type: {cfg.loss_type}")
    print(f"Device: {device}")
    print(f"Bootstrap samples: {cfg.n_boot}")
    
    for disease in cfg.diseases:
        # Load data
        try:
            (train_loader, val_loader, test_loader,
             X_train, y_train, train_ids, val_ids, test_ids) = create_dataloaders(
                cfg.data_root, disease, cfg.batch_size, device
            )
        except FileNotFoundError as e:
            print(f"\nERROR: Could not load data for {disease}")
            print(f"  {e}")
            continue
        
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1] if y_train.ndim == 2 else 1
        
        # Initialize model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        model = FlexibleMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        
        # Setup loss function
        pos_counts, neg_counts = compute_case_control_counts(y_train)
        loss_fn = DynamicWeightedLoss(
            loss_type=cfg.loss_type,
            f_case=pos_counts,
            f_control=neg_counts,
            device=device,
        )
        
        # Train and evaluate
        test_ids_np = None if test_ids is None else (
            test_ids.cpu().numpy() if torch.is_tensor(test_ids) else np.asarray(test_ids)
        )
        val_ids_np = None if val_ids is None else (
            val_ids.cpu().numpy() if torch.is_tensor(val_ids) else np.asarray(val_ids)
        )
        
        metrics, optimal_threshold = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_epochs=cfg.epochs,
            patience=cfg.patience,
            device=device,
            disease=disease,
            loss_name=cfg.loss_type,
            output_dir=cfg.output_dir,
            n_boot=cfg.n_boot,
            ci=cfg.ci,
            stratified_boot=cfg.stratified_boot,
            test_ids=test_ids_np,
            val_ids=val_ids_np,
        )
        
        # Save model if requested
        if cfg.save_model:
            disease_dir = os.path.join(cfg.output_dir, disease)
            os.makedirs(disease_dir, exist_ok=True)
            model_path = os.path.join(disease_dir, f"best_model_{cfg.loss_type}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model: best_model_{cfg.loss_type}.pth")
        
        # Store results
        row = {"disease": disease, "loss_type": cfg.loss_type, **metrics}
        results_rows.append(row)
        
        # Save per-disease results
        disease_df = pd.DataFrame([row])
        disease_dir = os.path.join(cfg.output_dir, disease)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disease_csv = os.path.join(disease_dir, f"{disease}_metrics.csv")
        disease_df.to_csv(disease_csv, index=False)
        print(f"  Saved metrics: {os.path.basename(disease_csv)}")
    
    # ==================== Save Aggregate Results ====================
    if not results_rows:
        print("\nWARNING: No results generated!")
        return pd.DataFrame()
    
    all_df = pd.DataFrame(results_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_csv = os.path.join(cfg.output_dir, f"all_diseases_metrics.csv")
    all_df.to_csv(all_csv, index=False)
    
    print(f"\n{'='*80}")
    print("Pipeline Complete!")
    print(f"{'='*80}")
    print(f"\nAggregate results saved to: {all_csv}")
    print(f"\nPer-disease outputs in: {cfg.output_dir}/<disease>/")
    print(f"  - Participant predictions: <disease>_<loss>_predictions_*.csv")
    print(f"  - ROC curve data: <disease>_<loss>_roc_curve_*.csv")
    print(f"  - Metrics summary: <disease>_metrics_*.csv")
    if cfg.save_model:
        print(f"  - Trained model: best_model_{cfg.loss_type}.pth")
    
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    for _, row in all_df.iterrows():
        print(f"\n{row['disease']}:")
        print(f"  ROC-AUC: {row.get('roc_auc_ci_formatted', 'N/A')}")
        print(f"  PR-AUC:  {row.get('pr_auc_ci_formatted', 'N/A')}")
        print(f"  Brier:   {row.get('brier_ci_formatted', 'N/A')}")
        print(f"  MCC:     {row.get('mcc_ci_formatted', 'N/A')}")
        print(f"  F1:      {row.get('f1_score_ci_formatted', 'N/A')}")
        print(f"  Threshold: {row.get('threshold', 'N/A'):.3f}")
    
    return all_df


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Unified disease prediction training pipeline"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing {disease}/{disease}_{split}.pt files"
    )
    parser.add_argument(
        "--diseases", type=str, required=True,
        help="Comma-separated disease names (e.g., 'afib,diabetes,cvd')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save results and models"
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save_model", action="store_true", help="Save trained models")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"]
    )
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level")
    parser.add_argument(
        "--no_stratified_boot", action="store_true",
        help="Disable stratified bootstrap"
    )
    
    args = parser.parse_args()
    diseases = [d.strip() for d in args.diseases.split(",") if d.strip()]
    
    return TrainConfig(
        data_root=args.data_root,
        diseases=diseases,
        output_dir=args.output_dir, 
        device=args.device,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        save_model=args.save_model,
        loss_type="weighted_bce",
        n_boot=args.n_boot,
        ci=args.ci,
        stratified_boot=(not args.no_stratified_boot),
    )


def main():
    cfg = parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()