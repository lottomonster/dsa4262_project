import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
if "__file__" in globals():
    script_dir = os.path.dirname(__file__)
else:
    # Fallback: assume current working directory
    script_dir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
import random
import argparse
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc


# =========================
# Default config
# =========================
DEFAULT_CONFIG = {
    "input_file": "../data/processed/dataset0_processed.parquet",
    "output_path": "../data/results",
    "rng": 42,
    "split_ratios": {"Train": 0.85, "Test": 0.15},
    "monitor": "val_pr_auc",                            # val_pr_auc, val_roc_auc, val_roc_plus_pr
    "patience": 20,                                     # this is the number of epoch where monitor does not improve before early stopping kicks in
    "k_fold_splits": 5,                                 # set this to 0 or 1 to skip k_fold modelling
}

print("List of GPUs: ", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# =========================
# Metric class for combined AUC
# =========================
class CombinedAUC(tf.keras.metrics.Metric):
    def __init__(self, name="combined_auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.roc_auc = tf.keras.metrics.AUC(curve="ROC", name="roc_auc")
        self.pr_auc = tf.keras.metrics.AUC(curve="PR", name="pr_auc")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.roc_auc.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.pr_auc.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.roc_auc.result() + self.pr_auc.result()

    def reset_state(self):
        self.roc_auc.reset_state()
        self.pr_auc.reset_state()


# =========================
# Build model
# =========================
def build_model(input_dim):
    tf.random.set_seed(42)

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            CombinedAUC(name="roc_plus_pr"),
        ],
    )
    return model


# =========================
# Dataset splitting
# =========================
def assign_train_test_split(reads_df, split_ratios={'Train': 0.85, 'Test': 0.15}, random_state=42):
    """
    Assign each row in reads_df a 'set_type' of Train or Test.
    Ensures:
        - All rows with the same gene_id are in the same set
        - Total number of rows in each set matches split_ratios closely
        - Label balance is approximately preserved
    """

    # Step 1: Compute per-gene statistics
    gene_stats = (
        reads_df
        .groupby('gene_id')['label']
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: 'label_0', 1: 'label_1'})
        .reset_index()
    )
    gene_stats['total'] = gene_stats['label_0'] + gene_stats['label_1']

    # Shuffle genes
    gene_stats = gene_stats.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Step 2: Compute total targets
    total_rows = gene_stats['total'].sum()
    target_rows = {k: total_rows * v for k, v in split_ratios.items()}

    # Step 3: Initialize bins for Train and Test sets
    bins = {k: {'genes': [], 'total': 0, 'label_0': 0, 'label_1': 0} for k in split_ratios}

    # Step 4: Sort genes largest → smallest to improve fit
    gene_stats = gene_stats.sort_values('total', ascending=False).reset_index(drop=True)

    # Step 5: Greedy assignment to keep splits close to target
    for _, row in gene_stats.iterrows():
        # Calculate deficit for each bin
        deficits = {k: target_rows[k] - bins[k]['total'] for k in bins}
        # Choose bin with largest positive deficit (most underfilled)
        chosen_bin = max(deficits, key=deficits.get)

        bins[chosen_bin]['genes'].append(row['gene_id'])
        bins[chosen_bin]['total'] += row['total']
        bins[chosen_bin]['label_0'] += row['label_0']
        bins[chosen_bin]['label_1'] += row['label_1']

    # Step 6: Map genes to set_type
    gene_to_set = {gene_id: set_name for set_name, data in bins.items() for gene_id in data['genes']}
    reads_df['set_type'] = reads_df['gene_id'].map(gene_to_set)

    # Step 7: Create train and test datasets
    train_df = reads_df[reads_df['set_type'] == 'Train']
    test_df = reads_df[reads_df['set_type'] == 'Test']

    # Step 8: Print achieved ratios
    actual_counts = reads_df['set_type'].value_counts(normalize=True).to_dict()
    print("Dataset sizes:")
    print(f"  - Train: {len(train_df):,} rows")
    print(f"  - Test: {len(test_df):,} rows")

    print("\nLabel distribution:")
    for name, df in zip(["Train", "Test"], [train_df, test_df]):
        dist = df['label'].value_counts(normalize=True)
        print(f"  - {name}: label 0 = {dist.get(0, 0)*100:.2f}%, label 1 = {dist.get(1, 0)*100:.2f}%")

    return train_df, test_df


# =========================
# Fold assignment
# =========================
def assign_fold_split(train_df, n_splits=5, random_state=42):
    """
    Assign each gene to a fold (0 .. n_splits-1) for cross-validation.
    Ensures all rows of a gene are in the same fold and roughly preserves label ratios.
    """

    # Just go away SettingWithCopyError
    train_df = train_df.copy()

    # Compute per-gene stats
    gene_stats = (
        train_df.groupby("gene_id")["label"]
        .mean()
        .reset_index()
    )
    gene_stats = gene_stats.sample(frac=1, random_state=random_state)

    # Assign folds greedily
    folds = {i: {"genes": [], "label_sum": 0, "n_genes": 0} for i in range(n_splits)}
    for _, row in gene_stats.iterrows():
        # Choose fold with minimal cumulative label sum
        fold_idx = min(folds, key=lambda f: folds[f]["label_sum"])
        folds[fold_idx]["genes"].append(row["gene_id"])
        folds[fold_idx]["label_sum"] += row["label"]
        folds[fold_idx]["n_genes"] += 1

    # Map gene_id → fold
    gene_to_fold = {gene: f for f, data in folds.items() for gene in data["genes"]}
    train_df.loc[:, "fold"] = train_df["gene_id"].map(gene_to_fold)
    
    # Print fold statistics
    print("\n===== Fold Assignment Summary =====")
    for fold in range(n_splits):
        fold_df = train_df[train_df["fold"] == fold]
        n_rows = len(fold_df)
        count_0 = (fold_df['label'] == 0).sum()
        count_1 = (fold_df['label'] == 1).sum()
        pct_0 = count_0 / n_rows * 100
        pct_1 = count_1 / n_rows * 100

        print(f"\nFold {fold}:")
        print(f"  - Total rows: {n_rows:,}")
        print(f"  - Label 0: {count_0:,} ({pct_0:.2f}%)")
        print(f"  - Label 1: {count_1:,} ({pct_1:.2f}%)")
    
    return train_df


# =========================
# Train and evaluate
# =========================
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, config, base, output_path, extra_name = "_", save = True, verbose = 1):

    start_time = time.time()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, cw))

    early_stop = EarlyStopping(
        monitor=config['monitor'], mode="max", patience=config['patience'], restore_best_weights=True
    )

    model = build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stop],
        verbose=verbose,
    )


    # Predictions
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    report = classification_report(y_test, y_pred, digits=4)

    # Get the time taken for this fold
    end_time = time.time()
    duration = end_time - start_time

    print(report)
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC : {pr_auc:.4f}")
    print(f"Time taken to train: {duration}")

    if save:
        # Save metrics
        metrics_file = output_path / f"{base}{extra_name}metrics.txt"
        with open(metrics_file, "w") as f:
            f.write(report)
            f.write(f"\nTest ROC-AUC: {roc_auc:.4f}\n")
            f.write(f"Test PR-AUC : {pr_auc:.4f}\n")

        # Plot PR curve
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall Curve (Test Set)")
        plt.tight_layout()
        plt.savefig(output_path / f"{base}{extra_name}precision_recall_curve.png", dpi=300)
        plt.close()

    return model, scaler, history

def k_fold_train(train_df, config, base, output_path):
    """
    Perform k-fold cross-validation on train_df.
    Returns: list of trained models, scalers, fold metrics
    """
    n_splits = config.get("k_fold_splits", 5)

    # Assign folds to genes
    train_df = assign_fold_split(train_df, n_splits=n_splits, random_state=config['rng'])

    # fold_models = []
    # fold_scalers = []
    fold_metrics = []

    for fold in range(n_splits):

        fold_train_df = train_df[train_df["fold"] != fold]
        fold_val_df   = train_df[train_df["fold"] == fold]

        X_train, y_train = fold_train_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "set_type", "label", "fold"], errors="ignore"), fold_train_df["label"]
        X_val, y_val     = fold_val_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "set_type", "label", "fold"], errors="ignore"), fold_val_df["label"]

        # Here we use the same X_val as "test" to get fold metrics
        model, scaler, history = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_val, y_val, config,
            base, output_path, save = False, verbose = 0,
        )

        # Compute metrics
        y_pred_prob = model.predict(scaler.transform(X_val))
        roc = roc_auc_score(y_val, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
        pr = auc(recall, precision)
  

        n_epochs_ran = len(history.history['loss'])
        print(f"Fold {fold}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, Epochs ran: {n_epochs_ran}")

    # Summarize
    mean_roc = np.mean([m["roc_auc"] for m in fold_metrics])
    std_roc = np.std([m["roc_auc"] for m in fold_metrics])
    mean_pr = np.mean([m["pr_auc"] for m in fold_metrics])
    std_pr = np.std([m["pr_auc"] for m in fold_metrics])

    summary_file = output_path / f"{base}_{n_splits}_fold_summary.txt"
    with open(summary_file, "w") as f:
        for m in fold_metrics:
            f.write(f"Fold {m['fold']}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}\n")
        f.write(f"\nMean ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f}\n")
        f.write(f"Mean PR-AUC : {mean_pr:.4f} ± {std_pr:.4f}\n")

    print(f"K-Fold summary saved to: {summary_file}")

    return fold_metrics

def plot_and_save_models(model, scaler, history, output_path, base, extra_name="_"):
    """
    Plot ROC-AUC and PR-AUC curves and save them along with model and scaler.
    
    Parameters:
    - history: Keras History object
    - scaler: fitted scaler object
    - model: trained Keras model
    - output_path: Path to save files
    - base: base filename prefix
    - extra_name: string to insert in saved file names
    """

    # Plot ROC-AUC curve
    print(f"Plotting ROC-AUC curve ({extra_name})...")
    plt.figure(figsize=(10,5))
    plt.plot(history.history['roc_auc'], label='Train ROC-AUC')
    if 'val_roc_auc' in history.history:
        plt.plot(history.history['val_roc_auc'], label='Val ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.title(f'Training vs Validation ROC-AUC ({extra_name})')
    plt.tight_layout()
    plt.savefig(output_path / f"{base}_{extra_name}_roc_auc_curve.png", dpi=300)
    plt.close()
    print(f"ROC-AUC curve ({extra_name}) saved")

    # Plot PR-AUC curve
    print(f"Plotting PR-AUC curve ({extra_name})...")
    plt.figure(figsize=(10,5))
    plt.plot(history.history['pr_auc'], label='Train PR-AUC')
    if 'val_pr_auc' in history.history:
        plt.plot(history.history['val_pr_auc'], label='Val PR-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('PR-AUC')
    plt.legend()
    plt.title(f'Training vs Validation PR-AUC ({extra_name})')
    plt.tight_layout()
    plt.savefig(output_path / f"{base}_{extra_name}_pr_auc_curve.png", dpi=300)
    plt.close()
    print(f"PR-AUC curve ({extra_name}) saved")

    # Save model and scaler
    joblib.dump(scaler, output_path / f"{base}_{extra_name}_scaler.pkl")
    model.save(output_path / f"{base}_{extra_name}_model.keras")
    print(f"Model and scaler ({extra_name}) saved to {output_path}")

# =========================
# Main function
# =========================
def main(config=None):
    if config is None:
        config = DEFAULT_CONFIG

    # Set seeds
    random_state = config['rng']
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    print(f"Random seeds set to {random_state}")

    # Paths
    input_file = (Path(script_dir) / Path(config.get("input_file"))).resolve()
    output_path = (Path(script_dir) / Path(config.get("output_path"))).resolve()
    os.makedirs(output_path, exist_ok=True)
    base = input_file.stem
    if "_processed" in base:
        base = base.replace("_processed", "")
    print(f"Input file: {input_file}")
    print(f"Output path: {output_path}")
    print(f"Base name for outputs: {base}")

    # Load data
    df = pd.read_parquet(input_file)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

    # Train/test split
    print("Performing train/test split...")
    trainval_df, test_df = assign_train_test_split(df, config['split_ratios'], random_state)

    # K-fold cross-validation, Split and Train 
    if config['k_fold_splits'] > 1:
        print(f"Starting {config['k_fold_splits']}-fold cross-validation...")
        fold_metrics = k_fold_train(trainval_df, config, base=base, output_path=output_path)
        print("K-fold cross-validation completed")

    # Train and Validate on Training set, Test on Testing
    print("Training and Validating model on train set and evaluating on test set...")
    X_trainval_df = trainval_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_trainval_df = trainval_df["label"]
    X_test = test_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_test = test_df["label"]

    model, scaler, history = train_and_evaluate(X_trainval_df, y_trainval_df, X_trainval_df, y_trainval_df, X_test, y_test, config, base, output_path, extra_name= "_test_split_", save = True)
    plot_and_save_models(model, scaler, history, output_path, base, extra_name="test_split")
    print("Training and evaluation completed for training/validation vs test")

    # Full train/validation set
    print("Performing full train/validation split (95/5), train model on train split(95) and validating on validation split(5)...")
    full_train_df, full_val_df = assign_train_test_split(df, {"Train": 0.95, "Test": 0.05}, random_state)
    X_full_train = full_train_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_full_train = full_train_df["label"]
    X_full_val = full_val_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_full_val = full_val_df["label"]

    model, scaler, history = train_and_evaluate(X_full_train, y_full_train, X_full_val, y_full_val, X_full_val, y_full_val, config, base, output_path, extra_name= "_full_", save = True)
    plot_and_save_models(model, scaler, history, output_path, base, extra_name="full")
    print("Full training completed")

# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or inference a model")
    parser.add_argument("--input_file", type=str, help="Path to input parquet file")
    parser.add_argument("--output_path", type=str, help="Output directory")
    parser.add_argument("--rng", type=int, help="Random seed")
    parser.add_argument("--split_ratios", type=str, help="Train/Test split as JSON, e.g., '{\"Train\":0.8,\"Test\":0.2}'")
    parser.add_argument("--monitor", type=str, choices = ['val_pr_auc', 'val_roc_auc', 'val_roc_plus_pr'], help="Validation Metric used for early stopping")
    parser.add_argument("--patience", type=int, help="Number of epoch where there is no improvement in monitor before early stopping")
    parser.add_argument("--k_fold_splits", type=int, help="Number of k-fold, set this this 0 or 1 to skip k-fold validation")
    args = parser.parse_args()

    # Merge config
    config = {**DEFAULT_CONFIG, **{k: (json.loads(v) if k == "split_ratios" else v) 
                                for k, v in vars(args).items() if v is not None}}
    main(config)
