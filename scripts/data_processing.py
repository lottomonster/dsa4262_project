# =========================
# 0a. Imports & Setup
# =========================
import os
import argparse
from pathlib import Path
import random

import numpy as np
import polars as pl

# Handle path setup (so it works when run from anywhere)
if "__file__" in globals():
    script_dir = os.path.dirname(__file__)
else:
    script_dir = os.getcwd()

# =========================
# 0b. Default Configuration
# =========================
DEFAULT_CONFIG = {
    "input_file": "../data/csv/dataset0.csv",
    "output_path": "../data/processed/",
    "rng": 42,
    "mode": "train",  # "train" or "inference"
}

# =========================
# Main function
# =========================
def main(config=None):
    mode = config.get("mode", "train").lower()
    rng = config.get("rng", 42)
    np.random.seed(rng)
    random.seed(rng)

    # File paths
    input_file = (Path(script_dir) / Path(config.get("input_file"))).resolve()
    output_path = (Path(script_dir) / Path(config.get("output_path"))).resolve()
    base = input_file.stem
    output_file = output_path / f"{base}_processed.parquet"

    print(f"\nData Processing ({mode.upper()} mode)")
    print(f"Input file: {input_file}")

    # =========================
    # 1. Load CSV with Polars
    # =========================
    numeric_cols = [
        'PreTime', 'PreSD', 'PreMean',
        'InTime', 'InSD', 'InMean',
        'PostTime', 'PostSD', 'PostMean'
    ]

    schema_overrides = {col: pl.Float64 for col in numeric_cols}
    df = pl.read_csv(input_file, schema_overrides=schema_overrides)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0")


    # =========================
    # 2. Aggregation
    # =========================
    print("Performing aggregation...")
    if mode == "train":
        required_cols = {"gene_id", "label"}
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for training mode: {missing}")
        group_cols = ["gene_id", "label", "transcript_id", "transcript_position", "7mer"]
    else:
        group_cols = ["transcript_id", "transcript_position", "7mer"]

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

    # =========================
    # 3. Vectorized One-hot 7-mer encoding
    # =========================
    print("Encoding 7-mer sequences (vectorized)...")
    seq_array = np.array([list(s) for s in df_agg["7mer"].to_numpy()])
    mapping = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])  # A,C,G,T,others
    char_to_idx = {b:i for i,b in enumerate("ACGT")}

    idx_array = np.vectorize(lambda c: char_to_idx.get(c,4))(seq_array)
    one_hot = mapping[idx_array].reshape(len(df_agg), 28)  # 7 positions * 4 bases

    one_hot_cols = [f"pos{i}_{b}" for i in range(7) for b in "ACGT"]
    one_hot_df = pl.DataFrame({name: one_hot[:, i] for i, name in enumerate(one_hot_cols)})

    # =========================
    # 4. Vectorized sequence-based binary features
    # =========================
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

    # =========================
    # 5. Combine everything
    # =========================
    df_final = pl.concat([df_agg, one_hot_df, df_features], how="horizontal")
    df_final = df_final.drop("7mer")

    # =========================
    # 6. Save as Parquet
    # =========================
    os.makedirs(output_path, exist_ok=True)
    df_final.write_parquet(output_file)
    print(f"Saved processed data to: {output_file}")
    print("Data Processing complete.\n")

    return df_final

# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing")
    parser.add_argument("--mode", type=str, choices=["train", "inference"],
                        help=f"Run mode (default: {DEFAULT_CONFIG['mode']})")
    parser.add_argument("--input_file", type=str,
                        help=f"Path to input CSV (default: {DEFAULT_CONFIG['input_file']})")
    parser.add_argument("--output_path", type=str,
                        help=f"Output directory (default: {DEFAULT_CONFIG['output_path']})")
    parser.add_argument("--rng", type=int,
                        help=f"Random seed (default: {DEFAULT_CONFIG['rng']})")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **{k:v for k,v in vars(args).items() if v is not None}}
    main(config)
