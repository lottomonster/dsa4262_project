import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import random
import argparse
from pathlib import Path
import time

import pandas as pd
import numpy as np
import joblib

import tensorflow as tf

# =========================
# Default Config for Inference
# =========================
DEFAULT_CONFIG = {
    "model_file": "../data/results/dataset0_full_model.keras",  # Path to the saved model
    "scaler_file": "../data/results/dataset0_full_scaler.pkl",  # Path to the saved scaler
    "input_file": "../data/processed/test_dataset_processed.parquet",  # Path to input data for inference
    "output_path": "../data/inference",  # Path to save the results
    "rng": 42,
    "full_keep" : False # If true it will also save a full dataset + a new column of score, else it will only be a csv that contains the 3 columns necessary for submission
}


# =========================
# Inference Function
# =========================
def make_inference(model, scaler, input_df, output_path, base='', full_keep_df=DEFAULT_CONFIG['full_keep']):
    """
    Use the trained model to make predictions on the input data
    and save the results to a CSV file.
    """

    # Preprocess the input data (same steps as training)
    X_input = input_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    X_input_scaled = scaler.transform(X_input)
    nan_mask = X_input.isna().any(axis=1)  # rows with any NaN, im looking at you sgnex data why so many groups with single row, or cant be aggregated properly

    # Make predictions
    y_pred_prob = model.predict(X_input_scaled)

    # Prepare the 3 columns required for submission
    output_df = input_df[["transcript_id", "transcript_position"]].copy()
    output_df["score"] = y_pred_prob  # Inference probability as the score
    output_df.loc[nan_mask, "score"] = np.nan   # manually set NaN for those rows, for different machine where defulat NA return a value instead
    output_file = output_path / f"{base}_inference_results.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")
    
    # Merge with full dataset if full_keep_df is True
    if full_keep_df:
        full_output_df = input_df.copy()
        full_output_df["score"] = y_pred_prob
        full_output_df.loc[nan_mask, "score"] = np.nan   # manually set NaN for those rows, for different machine where defulat NA return a value instead
        full_output_file = output_path / f"{base}_full_inference_results.csv"
        full_output_df.to_csv(full_output_file, index=False)
        print(f"\nInference results (FULL) saved to {full_output_file}")
    
    return output_df

# Very impt messages, do not delete
messages = [
    "Inference completed! Thanks for your patience – hope the rest of your day goes smoothly! ✨",
    "Inference completed! Appreciate you waiting – wishing you a chill and productive rest of your day! 🌸",
    "Inference completed! Thanks for sticking with it – hope the rest of your day is stress-free! 🌿",
    "Inference completed! We got there – sending good vibes for the rest of your day! 🌈",
    "Inference completed! Thanks for your patience – hope you crush whatever's next on your agenda! 💼",
    "Inference completed! Appreciate the wait – wishing you a vibe-filled rest of the day! ✌️",
    "Inference completed! Thanks for hanging in there – may the rest of your day be easy and rewarding! 🙌",
    "Inference completed! You made it – hope the rest of your day is smooth sailing! 🚤",
    "Inference completed! Thanks for your time – here’s to a productive and stress-free rest of your day! 💫",
    "Inference completed! Appreciate you – hope your day just gets better from here! 💙"
]


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

    # Model file
    model_file = Path(config["model_file"])
    if not model_file.is_absolute():
        model_file = (Path(script_dir) / Path(config["model_file"])).resolve()

    # Scaler file
    scaler_file = Path(config["scaler_file"])
    if not scaler_file.is_absolute():
        scaler_file = (Path(script_dir) / Path(config["scaler_file"])).resolve()

    # Determine base name for outputs
    base = input_file.stem
    if "_processed" in base:
        base = base.replace("_processed", "")

    print(f"\nInput file: {input_file}")
    print(f"\nOutput path: {output_path}")
    print(f"\nBase name for inference: {base}")

    # Return resolved paths and base name
    return input_file, output_path, model_file, scaler_file, base


# =========================
# Main function for Inference
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

    # Set paths
    input_file, output_path, model_file, scaler_file, base = resolve_paths(config)
 
    # Load the model and scaler
    print(f"\nLoading model from {model_file}...")
    tf.keras.backend.set_floatx('float32')
    model = tf.keras.models.load_model(model_file)

    print(f"\nLoading scaler from {scaler_file}...")
    scaler = joblib.load(scaler_file)

    # Load the input data for inference
    print(f"\nLoading input data from {input_file}...")
    input_df = pd.read_parquet(input_file)
    print(f"\nData loaded: {len(input_df)} rows, {len(input_df.columns)} columns")

    # Perform inference
    print("\nMaking predictions...")
    make_inference(model, scaler, input_df, output_path, base, full_keep_df=config['full_keep'])

    random.seed(time.time())
    print("\n" + random.choice(messages) + "\n")


# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make inference using a trained model")
    parser.add_argument("--model_file", type=str, 
        help=f"Path to the trained model. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['model_file']})")
    parser.add_argument("--scaler_file", type=str, 
        help=f"Path to the saved scaler. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['scaler_file']})")
    parser.add_argument("--input_file", type=str, 
        help=f"Path to input data for inference. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['input_file']})")
    parser.add_argument("--output_path", type=str, 
        help=f"Path to save inference results. Take note it is relative to where this script is stored (default: {DEFAULT_CONFIG['output_path']})")
    parser.add_argument("--rng", type=int, 
        help=f"Random seed (default: {DEFAULT_CONFIG['rng']})")
    parser.add_argument("--full_keep", type=str, choices=["True", "False", "true", "false"],
                        help=f"If set to True, it will full input dataframe + score (default: {DEFAULT_CONFIG['full_keep']})")
    args = parser.parse_args()

    BOOL_KEYS = {"full_keep"}

    config = DEFAULT_CONFIG.copy()

    # Merge config
    for k, v in vars(args).items():
        if v is None:
            continue
        elif k in BOOL_KEYS and isinstance(v, str):
            v = (v.lower() == "true")
        config[k] = v

    main(config)
