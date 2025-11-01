import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
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
# Default Config for Training
# =========================
DEFAULT_CONFIG = {
    "input_file": "../data/processed/dataset0_processed.parquet",
    "output_path": "../data/results",
    "rng": 42,
    "split_ratio": {"Train": 0.85, "Test": 0.15},       # user will just input 2 float number that should sum to 1
    "monitor": "val_pr_auc",                            # val_pr_auc, val_roc_auc
    "patience": 20,                                     # this is the number of epoch where monitor does not improve before early stopping kicks in
    "k_fold_splits": 5,                                 # if this is smaller than 2 it will skip k-fold validation even if run_kfold is True
    "train_val_ratio" : {"Train": 0.95, "Test": 0.05} , # user will just input 2 float number that should sum to 1
    "run_kfold": False,                                 # whether to perform k-fold validation
    "run_full_model": True,                             # whether to train and validate model based on train_val_ratio, all of the dataset to participate in the training
}

print("List of GPUs: ", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# =========================
# Build model
# =========================
def build_model(input_dim):

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
            tf.keras.metrics.AUC(curve="PR", name="pr_auc")
        ],
    )
    return model


# =========================
# Anti Intrusive Thoughts
# =========================
def validate_ratios(ratio_dict, name):
    if not isinstance(ratio_dict, dict):   # this check unlikely required after modifications to arguments passed
        raise ValueError(f"{name} must be a JSON object like {{'Train': 0.8, 'Test': 0.2}}.")

    if set(ratio_dict.keys()) != {"Train", "Test"}:     # this check unlikely required after modifications to arguments passed
        raise ValueError(f"{name} must have exactly 'Train' and 'Test' keys (current keys: {list(ratio_dict.keys())})")

    if not all(isinstance(v, (int, float)) for v in ratio_dict.values()):
        raise ValueError(f"All values in {name} must be numbers.")

    if any(v <= 0 for v in ratio_dict.values()):
        raise ValueError(f"All values in {name} must be positive and greater than zero.")

    total = sum(ratio_dict.values())        # this check unlikely as similar check was performed before 
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"The values in {name} must sum to 1.0 (current total = {total}).")

def validate_kfold(run_kfold, splits):
    if run_kfold and (not isinstance(splits, int) or splits < 2):
        raise ValueError(f"k_fold_splits must be >= 2 when run_kfold=True. Got {splits}.")
    if not run_kfold and splits >= 2:
        print("[Warning] k_fold_splits provided but run_kfold=False. Ignoring.")

def validate_monitor(monitor):
    if monitor not in ["val_pr_auc", "val_roc_auc"]:
        raise ValueError(f"Invalid monitor: {monitor}")

def validate_patience(patience):
    if not isinstance(patience, int) or patience < 1:
        raise ValueError(f"Patience must be positive integer, got {patience}")


# =========================
# Path Function
# =========================
def resolve_paths(config):
    # Determine script directory
    if "__file__" in globals():
        script_dir = Path(__file__).parent
    else:
        script_dir = Path.cwd()

    # Input file
    input_file = Path(config["input_file"])
    if not input_file.is_absolute():
        input_file = (Path(script_dir) / Path(config["input_file"])).resolve()

    # Output path
    output_path = Path(config["output_path"])
    if not output_path.is_absolute():
        output_path = (Path(script_dir) / Path(config["output_path"])).resolve()
    os.makedirs(output_path, exist_ok=True)

    # Determine base name for outputs
    base = input_file.stem
    if "_processed" in base:
        base = base.replace("_processed", "")

    print(f"\nInput file: {input_file}")
    print(f"\nOutput path: {output_path}")
    print(f"\nBase name for outputs: {base}")

    return input_file, output_path, base


# =========================
# Dataset splitting
# =========================
def assign_train_test_split(reads_df, split_ratio=DEFAULT_CONFIG['split_ratio'], random_state=DEFAULT_CONFIG['rng']):
    """
    Assign each row in reads_df a 'set_type' of Train or Test.
    Ensures:
        - All rows with the same gene_id are in the same set
        - Total number of rows in each set matches split_ratio closely
        - Label balance is approximately preserved
    """
    # Go away SettingWithCopyError
    reads_df = reads_df.copy()

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

    # Compute total targets
    total_rows = gene_stats['total'].sum()
    target_rows = {k: total_rows * v for k, v in split_ratio.items()}

    # nitialize bins for Train and Test sets
    bins = {k: {'genes': [], 'total': 0, 'label_0': 0, 'label_1': 0} for k in split_ratio}

    # Sort genes largest → smallest to improve fit
    gene_stats = gene_stats.sort_values('total', ascending=False).reset_index(drop=True)

    # Greedy assignment to keep splits close to target
    for _, row in gene_stats.iterrows():
        # Calculate deficit for each bin
        deficits = {k: target_rows[k] - bins[k]['total'] for k in bins}
        # Choose bin with largest positive deficit (most underfilled)
        chosen_bin = max(deficits, key=deficits.get)

        bins[chosen_bin]['genes'].append(row['gene_id'])
        bins[chosen_bin]['total'] += row['total']
        bins[chosen_bin]['label_0'] += row['label_0']
        bins[chosen_bin]['label_1'] += row['label_1']

    # Map genes to set_type
    gene_to_set = {gene_id: set_name for set_name, data in bins.items() for gene_id in data['genes']}
    reads_df['set_type'] = reads_df['gene_id'].map(gene_to_set)

    # Create train and test datasets
    train_df = reads_df[reads_df['set_type'] == 'Train']
    test_df = reads_df[reads_df['set_type'] == 'Test']

    # Print achieved ratio
    actual_counts = reads_df['set_type'].value_counts(normalize=True).to_dict()
    print("\nDataset sizes:")
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
def assign_fold_split(train_df, n_splits=DEFAULT_CONFIG['k_fold_splits'], random_state=DEFAULT_CONFIG['rng']):
    """
    Assign each gene to a fold (0 .. n_splits-1) for cross-validation.
    Ensures all rows of a gene are in the same fold and roughly preserves label ratio.
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

    # Map gene_id to fold
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
    n_splits = config['k_fold_splits']

    # Assign folds to genes
    train_df = assign_fold_split(train_df, n_splits=n_splits, random_state=config['rng'])

    fold_metrics = []

    # k-fold validation
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
        fold_metrics.append({"fold": fold, "roc_auc": roc, "pr_auc": pr, "epochs": n_epochs_ran})
        
        print(f"Fold {fold}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, Epochs ran: {n_epochs_ran}")

    # Summarise
    mean_roc = np.mean([m["roc_auc"] for m in fold_metrics])
    std_roc = np.std([m["roc_auc"] for m in fold_metrics])
    mean_pr = np.mean([m["pr_auc"] for m in fold_metrics])
    std_pr = np.std([m["pr_auc"] for m in fold_metrics])

    # Saving a k-fold summary results
    summary_file = output_path / f"{base}_{n_splits}_fold_summary.txt"
    with open(summary_file, "w") as f:
        for m in fold_metrics:
            f.write(f"Fold {m['fold']}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, epochs={m['epochs']}\n")
        f.write(f"\nMean ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f}\n")
        f.write(f"Mean PR-AUC : {mean_pr:.4f} ± {std_pr:.4f}\n")
    print(f"\nK-Fold summary saved to: {summary_file}")

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
    print(f"\nPlotting ROC-AUC curve ({extra_name})...")
    plt.figure(figsize=(10,5))
    plt.plot(history.history['roc_auc'], label='Train ROC-AUC')
    if 'val_roc_auc' in history.history:
        plt.plot(history.history['val_roc_auc'], label='Val ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.title(f'Training vs Validation ROC-AUC ({extra_name})')
    plt.tight_layout()
    plt.savefig(output_path / f"{base}_{extra_name}_training_vs_validation_roc_auc_curve.png", dpi=300)
    plt.close()
    print(f"ROC-AUC curve ({extra_name}) saved")

    # Plot PR-AUC curve
    print(f"\nPlotting PR-AUC curve ({extra_name})...")
    plt.figure(figsize=(10,5))
    plt.plot(history.history['pr_auc'], label='Train PR-AUC')
    if 'val_pr_auc' in history.history:
        plt.plot(history.history['val_pr_auc'], label='Val PR-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('PR-AUC')
    plt.legend()
    plt.title(f'Training vs Validation PR-AUC ({extra_name})')
    plt.tight_layout()
    plt.savefig(output_path / f"{base}_{extra_name}_training_vs_validation_pr_auc_curve.png", dpi=300)
    plt.close()
    print(f"PR-AUC curve ({extra_name}) saved")

    # Save model and scaler
    joblib.dump(scaler, output_path / f"{base}_{extra_name}_scaler.pkl")
    model.save(output_path / f"{base}_{extra_name}_model.keras")
    print(f"\nModel and scaler ({extra_name}) saved to {output_path}")

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
    print(f"\nRandom seeds set to {random_state}")

    # Set Paths
    input_file, output_path, base = resolve_paths(config)

    # Load data
    df = pd.read_parquet(input_file)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    print(f"\nData loaded: {len(df)} rows, {len(df.columns)} columns")

    # Train/test split
    print("Performing train/test split...")
    trainval_df, test_df = assign_train_test_split(df, config['split_ratio'], random_state)

    # K-fold cross-validation, Split and Train 
    if config['k_fold_splits'] > 1 and config['run_kfold']:
        print(f"\nStarting {config['k_fold_splits']}-fold cross-validation...")
        k_fold_train(trainval_df, config, base=base, output_path=output_path)
        print("\nK-fold cross-validation completed")


    train_num_split = config['split_ratio']['Train']*100
    test_num_split = config['split_ratio']['Test']*100
    train_test_split_name = f"{train_num_split}_{test_num_split}"

    # Training and Validating on Train set, Testing on Test set
    print(f"\nStarting model training: training on {train_num_split}% of the training data, validating on {test_num_split}% of the training data, and testing on {test_num_split}% of the entire dataset")
    train_df, val_df = assign_train_test_split(trainval_df, config['split_ratio'], random_state)
    X_train_df = train_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_train_df = train_df["label"]
    X_val_df = val_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_val_df = val_df["label"]
    X_test = test_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    y_test = test_df["label"]

    model, scaler, history = train_and_evaluate(X_train_df, y_train_df, X_val_df, y_val_df, X_test, y_test, config, base, output_path, extra_name= f"_{train_test_split_name}_", save = True)
    plot_and_save_models(model, scaler, history, output_path, base, extra_name=f"{train_test_split_name}")
    print(f"\nModel training complete")

    # Full train/validation set
    if config['run_full_model']:
        train_num_split = config['train_val_ratio']['Train']*100
        test_num_split = config['train_val_ratio']['Test']*100
        print(f"\nPerforming full train/validation split, training model on train split({train_num_split}%) and validating on validation split({test_num_split}%)...")
        full_train_df, full_val_df = assign_train_test_split(df, config['train_val_ratio'], random_state)
        X_full_train = full_train_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
        y_full_train = full_train_df["label"]
        X_full_val = full_val_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
        y_full_val = full_val_df["label"]

        model, scaler, history = train_and_evaluate(X_full_train, y_full_train, X_full_val, y_full_val, X_full_val, y_full_val, config, base, output_path, extra_name= "_full_", save = True)
        plot_and_save_models(model, scaler, history, output_path, base, extra_name="full")
        print("\nFull training completed")
    print("\nWe have reached the end of the training. Thank you for the wait. Have a nice day!")

# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="How to Train a Model")
    parser.add_argument("--input_file", type=str, 
                        help=f"Path to input parquet file. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['input_file']})")
    parser.add_argument("--output_path", type=str, 
                        help=f"Output directory. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['output_path']})")
    parser.add_argument("--rng", type=int, 
                        help=f"Random seed (default: {DEFAULT_CONFIG['rng']})")
    parser.add_argument("--split_ratio", type=float, nargs=2, metavar=('TRAIN', 'TEST'),
                        help=f"Train/Test split ratio as two numbers that sum to 1.0. Example: --split_ratio 0.8 0.2 (default: {json.dumps(DEFAULT_CONFIG['split_ratio'])})")
    parser.add_argument("--monitor", type=str, choices=['val_pr_auc', 'val_roc_auc'], 
                        help=f"Validation Metric used for early stopping (default: {DEFAULT_CONFIG['monitor']})")
    parser.add_argument("--patience", type=int, 
                        help=f"Number of epochs where there is no improvement in monitor before early stopping (default: {DEFAULT_CONFIG['patience']})")
    parser.add_argument("--run_kfold", type=str, choices=["True", "False", "true", "false"],
                        help=f"Enable or disable k-fold cross-validation (default: {DEFAULT_CONFIG['run_kfold']})")
    parser.add_argument("--k_fold_splits", type=int, 
                        help=f"Number of folds for k-fold cross-validation. Must be ≥ 2 to enable k-fold (default: {DEFAULT_CONFIG['k_fold_splits']})")
    parser.add_argument("--run_full_model", type=str, choices=["True", "False","true", "false"],
                        help=f"Train the final model on the full dataset (no test split). Use for final production training (default: {DEFAULT_CONFIG['run_full_model']})")
    parser.add_argument("--train_val_ratio", type=float, nargs=2, metavar=('TRAIN', 'VAL'),
                        help=f"Train/validation split ratio used when training the full model. Must sum to 1.0. Example: --train_val_ratio 0.9 0.1 \
                        (default: {json.dumps(DEFAULT_CONFIG['train_val_ratio'])})")
    args = parser.parse_args()

    BOOL_KEYS = {"run_kfold", "run_full_model"}
    JSON_KEYS = {"split_ratio", "train_val_ratio"}

    config = DEFAULT_CONFIG.copy()

    # Merge config
    for k, v in vars(args).items():
        if v is None:
            continue

        # Handle split_ratio and train_val_ratio
        if k in {"split_ratio", "train_val_ratio"} and isinstance(v, list):
            if abs(sum(v) - 1.0) > 1e-6:
                raise ValueError(f"{k} values must sum to 1. Got {v}")
            v = {"Train": v[0], "Test": v[1]} if k == "split_ratio" else {"Train": v[0], "Test": v[1]}

        # Convert string "True"/"False" to boolean only for keys in BOOL_KEYS
        elif k in BOOL_KEYS and isinstance(v, str):
            v = (v.lower() == "true")

        config[k] = v
    
    # In case someone has intrusive thoughts like me
    try:
        validate_ratios(config["split_ratio"], "split_ratio")
        validate_ratios(config["train_val_ratio"], "train_val_ratio")
        validate_kfold(config["run_kfold"], config["k_fold_splits"])
        validate_monitor(config["monitor"])
        validate_patience(config["patience"])
    except ValueError as e:
        print(f"[Config Error] {e}")
        sys.exit(1)

    main(config)