import os
import argparse
from pathlib import Path
import random
import time

import numpy as np
import polars as pl


# =========================
# Default Config for Data Processing
# =========================
DEFAULT_CONFIG = {
    "input_file": "../data/csv/test_dataset.csv",
    "output_path": "../data/processed/",
    "rng": 42,
    "mode": "inference",  # "train" or "inference"
}


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
        input_file = (script_dir / input_file).resolve()

    # Output path
    output_path = Path(config["output_path"])
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()
    os.makedirs(output_path, exist_ok=True)

    # Determine base name for outputs
    name = input_file.name
    if name.endswith((".csv.gz")):
        base = name.rsplit(".", 2)[0]  # remove both extensions
    else:
        base = input_file.stem  # remove single extension

    # Remove "_processed" suffix if present
    if "_processed" in base:
        base = base.replace("_processed", "")


    print(f"Input file: {input_file}")
    print(f"Output path: {output_path}")
    print(f"Base name for outputs: {base}")

    return input_file, output_path, base

# =========================
# Main function
# =========================
def main(config=None):
    if config is None:
        config = DEFAULT_CONFIG

    # Set seeds
    random_state = config['rng']
    np.random.seed(random_state)
    random.seed(random_state)
    print(f"\nRandom seeds set to {random_state}")

    # Training or Inference Mode
    mode = config['mode']

    # Set paths
    input_file, output_path, base = resolve_paths(config)
    output_file = output_path / f"{base}_processed.parquet"

    print(f"\nData Processing ({mode.upper()} mode)")
    print(f"Input file: {input_file}")

    numeric_cols = [
        'PreTime', 'PreSD', 'PreMean',
        'InTime', 'InSD', 'InMean',
        'PostTime', 'PostSD', 'PostMean'
    ]

    # MLP like float, they do not like whole
    schema_overrides = {col: pl.Float64 for col in numeric_cols}

    # Load CSV with Polar Bear
    df = pl.read_csv(input_file, schema_overrides=schema_overrides)

    # Just in case we have someone pass in a dataset that has this column
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0")

    # Aggregation
    print("\nPerforming aggregation...")
    if mode == "train":
        required_cols = {"gene_id", "label"}
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for training mode: {missing}")
        group_cols = ["gene_id", "label", "transcript_id", "transcript_position", "7mer"]
    else:
        group_cols = ["transcript_id", "transcript_position", "7mer"]

    # All the aggregation we apply
    agg_exprs = []
    for col in numeric_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).median().alias(f"{col}_median"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).quantile(0.25).alias(f"{col}_p25"),
            pl.col(col).quantile(0.75).alias(f"{col}_p75"),
            pl.col(col).skew().alias(f"{col}_skew"),
            pl.col(col).kurtosis().alias(f"{col}_kurtosis"),
            pl.col(col).mode().first().alias(f"{col}_mode")
        ])

    df_agg = df.group_by(group_cols).agg(agg_exprs)

    # Vectorized One-hot 7-mer encoding
    print("Encoding 7-mer sequences (vectorized)...")
    seq_array = np.array([list(s) for s in df_agg["7mer"].to_numpy()])
    mapping = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])  # A,C,G,T,others
    char_to_idx = {b:i for i,b in enumerate("ACGT")}

    idx_array = np.vectorize(lambda c: char_to_idx.get(c,4))(seq_array)
    one_hot = mapping[idx_array].reshape(len(df_agg), 28)  # 7 positions * 4 bases

    one_hot_cols = [f"pos{i}_{b}" for i in range(7) for b in "ACGT"]
    one_hot_df = pl.DataFrame({name: one_hot[:, i] for i, name in enumerate(one_hot_cols)})

    #  Vectorized important sequence-based binary features from our EDA
    print("Adding sequence-based binary features (vectorized)...")
    seq_strs = df_agg["7mer"].to_numpy()
    has_GGACT = np.array([s[1:6]=="GGACT" for s in seq_strs], dtype=int)
    has_GGA = np.array(['GGA' in s for s in seq_strs], dtype=int)
    has_AGG = np.array(['AGG' in s for s in seq_strs], dtype=int)
    has_TGG = np.array(['TGG' in s for s in seq_strs], dtype=int)
    has_TTA = np.array(['TTA' in s for s in seq_strs], dtype=int)
    has_GTA = np.array(['GTA' in s for s in seq_strs], dtype=int)
    has_TAA = np.array(['TAA' in s for s in seq_strs], dtype=int)

    df_features = pl.DataFrame({
        "has_GGACT": has_GGACT,
        "has_GGA": has_GGA,
        "has_AGG": has_AGG,
        "has_TGG": has_TGG,
        "has_TTA": has_TTA,
        "has_GTA": has_GTA,
        "has_TAA": has_TAA
    })

    # Combine everything
    df_final = pl.concat([df_agg, one_hot_df, df_features], how="horizontal")
    df_final = df_final.drop("7mer")

    # Save as Parquet
    os.makedirs(output_path, exist_ok=True)
    df_final.write_parquet(output_file)
    print(f"\nSaved processed data to: {output_file}")

    return df_final


# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Processing")
    parser.add_argument("--mode", type=str, choices=["train", "inference"],
                        help=f"Run mode (default: {DEFAULT_CONFIG['mode']})")
    parser.add_argument("--input_file", type=str,
                        help=f"Path to input CSV. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['input_file']})")
    parser.add_argument("--output_path", type=str,
                        help=f"Output directory. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['output_path']})")
    parser.add_argument("--rng", type=int,
                        help=f"Random seed (default: {DEFAULT_CONFIG['rng']})")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    # Merge config
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    main(config)
