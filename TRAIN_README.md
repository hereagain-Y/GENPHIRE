# Training Model with Embeddings

## Complete Pipeline in ONE Script

`train_model.py` does everything:
1. ✅ Loads embeddings from CSV
2. ✅ Creates simulated phenotype labels  
3. ✅ Randomly splits into train/val/test
4. ✅ Saves splits to disk
5. ✅ Trains MLP model
6. ✅ Evaluates on test set
7. ✅ Saves predictions & metrics

---

## Quick Start

### Assuming you have `toy_embeddings.csv`:

```bash
cd "/Users/hereagain/Library/CloudStorage/OneDrive-Emory/UK Biobank project - Documents/GENPHIRE"

# Run complete pipeline
python code/train_model.py \
    --input data/toy_embeddings.csv \
    --output_dir results \
    --phenotype simulated_disease \
    --epochs 100 \
    --batch_size 32
```

---

## What It Does

### Step 1: Loads Embeddings
- Reads `toy_embeddings.csv` (ID, sentence, embedding columns)
- Converts embedding strings to numpy arrays

### Step 2: Simulates Labels
- Creates binary labels (disease vs healthy)
- Default: 20% positive rate (realistic for diseases)
- **Note:** Replace with real phenotype data for actual research!

### Step 3: Splits Data
- Train: 70%
- Validation: 10%  
- Test: 20%
- Stratified split (maintains class balance)
- Saves as `.pt` files

### Step 4: Trains Model
- Simple MLP (Multi-Layer Perceptron)
- Early stopping (patience=20)
- BCE loss for binary classification

### Step 5: Evaluates
- ROC-AUC, F1, Precision, Recall
- Saves predictions & metrics

---

## Example Output

```
Using device: cpu

Loading embeddings from: data/toy_embeddings.csv
Loaded 201 samples with 1536 dimensions

Simulated binary labels: 42 positive, 159 negative

Splitting data (test=0.2, val=0.1)...
  Train: 140 samples
  Val:   21 samples
  Test:  40 samples

Training for up to 100 epochs (patience=20)...
  Epoch   1/100  Train Loss: 0.5234  Val Loss: 0.4987
  Epoch  11/100  Train Loss: 0.3456  Val Loss: 0.3892
  Early stopping at epoch 35
  Best validation loss: 0.3654

Evaluating model on test set...

============================================================
TEST SET RESULTS
============================================================
  Accuracy:  0.8750
  ROC-AUC:   0.8923
  F1 Score:  0.7273
  Precision: 0.8000
  Recall:    0.6667
============================================================

Saved predictions: results/simulated_disease/predictions_20251125_143052.csv
Saved metrics: results/simulated_disease/metrics_20251125_143052.csv

Pipeline Complete! ✓
```

---

## Output Files

After running, you'll have:

```
results/simulated_disease/
├── simulated_disease_train.pt          # Training data
├── simulated_disease_val.pt            # Validation data
├── simulated_disease_test.pt           # Test data
├── predictions_TIMESTAMP.csv           # Individual predictions
├── metrics_TIMESTAMP.csv               # Performance metrics
└── best_model.pth                      # Trained model (if --save_model)
```

### `predictions_TIMESTAMP.csv`:
```csv
ID,true_label,predicted_label,predicted_probability
6001036,0,0,0.1234
1863994,1,1,0.8765
...
```

### `metrics_TIMESTAMP.csv`:
```csv
accuracy,roc_auc,f1,precision,recall
0.8750,0.8923,0.7273,0.8000,0.6667
```

---

## All Options

```bash
python code/train_model.py \
    --input data/toy_embeddings.csv \      # Input embeddings CSV
    --output_dir results \                  # Output directory
    --phenotype diabetes \                  # Phenotype name
    --n_classes 2 \                        # Binary classification
    --batch_size 32 \                      # Batch size
    --hidden_dim 256 \                     # Hidden layer size
    --dropout 0.3 \                        # Dropout rate
    --lr 0.001 \                           # Learning rate
    --epochs 100 \                         # Max epochs
    --patience 20 \                        # Early stopping patience
    --test_size 0.2 \                      # Test set size
    --val_size 0.1 \                       # Validation size
    --seed 42 \                            # Random seed
    --save_model                           # Save trained model
```

---

## Using Real Phenotype Labels

To use **real** phenotype data instead of simulated:

### Option 1: Modify the script (line 68-77)

Replace:
```python
def simulate_phenotype_labels(n_samples, n_classes=2, seed=42):
    """Simulate phenotype labels."""
    np.random.seed(seed)
    labels = np.random.binomial(1, 0.2, n_samples)
    return labels
```

With:
```python
def load_real_phenotype_labels(label_file, id_col='ID', label_col='disease_status'):
    """Load real phenotype labels from CSV."""
    df = pd.read_csv(label_file)
    df = df.dropna(subset=[id_col, label_col])
    return df[id_col].values, df[label_col].astype(int).values
```

### Option 2: Provide label file as argument

Add to arguments:
```python
parser.add_argument('--labels', help='Path to phenotype labels CSV')
```

Then load:
```python
if args.labels:
    label_ids, y = load_real_phenotype_labels(args.labels)
    # Merge with embeddings by ID
else:
    y = simulate_phenotype_labels(len(X))
```

---

## Requirements

```bash
# Already in requirements.txt
pip install torch scikit-learn pandas numpy
```

---

## GPU Support

Automatically uses GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python code/train_model.py ...
```

---

## Tips

1. **Start small**: Test with `--epochs 20` first
2. **Tune hyperparameters**: Adjust `--hidden_dim`, `--dropout`, `--lr`
3. **Multiple runs**: Change `--seed` for different splits
4. **Save models**: Add `--save_model` for future predictions
5. **Monitor overfitting**: Check train vs val loss

---

## Next Steps

1. ✅ Run training pipeline
2. ✅ Check results in `results/` folder  
3. ✅ Visualize ROC curves (use predictions CSV)
4. ✅ Try different hyperparameters
5. ✅ Replace simulated labels with real phenotypes
6. ✅ Scale to full dataset

---

**Questions?** Check the script help:
```bash
python code/train_model.py --help
```

