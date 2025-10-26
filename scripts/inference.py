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

import pandas as pd
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler

# =========================
# Default Config for Inference
# =========================
DEFAULT_CONFIG = {
    "model_file": "../data/results/dataset0_full_model.keras",  # Path to the saved model
    "scaler_file": "../data/results/dataset0_full_scaler.pkl",  # Path to the saved scaler
    "input_file": "../data/processed/dataset0_processed.parquet",  # Path to input data for inference
    "output_path": "../data/inference",  # Path to save the results
    "rng": 42
}

# =========================
# Inference Function
# =========================
def make_inference(model, scaler, input_df, output_path, base = ''):
    """
    Use the trained model to make predictions on the input data
    and save the results to a CSV file.
    """

    # Preprocess the input data (same steps as training)
    X_input = input_df.drop(columns=["gene_id", "transcript_id", "transcript_position", "label", "set_type", "fold"], errors="ignore")
    X_input_scaled = scaler.transform(X_input)

    # Make predictions
    y_pred_prob = model.predict(X_input_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Prepare the output dataframe with only required columns
    output_df = input_df[["transcript_id", "transcript_position"]].copy()
    output_df["score"] = y_pred_prob  # Inference probability as the score
    
    # Save the results to a CSV file
    output_file = output_path / f"{base}_inference_results.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")

    return output_df

# =========================
# Metric class for combined AUC
# =========================
@register_keras_serializable(package="Custom")
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
    print(f"Random seeds set to {random_state}")

    # Set paths
    input_file = (Path(script_dir) / Path(config["input_file"])).resolve()
    output_path = (Path(script_dir) / Path(config["output_path"])).resolve()
    model_file = (Path(script_dir) / Path(config["model_file"])).resolve()
    scaler_file = (Path(script_dir) / Path(config["scaler_file"])).resolve()
    os.makedirs(output_path, exist_ok=True)
    base = input_file.stem
    if "_processed" in base:
        base = base.replace("_processed", "")
    print(f"Input file: {input_file}")
    print(f"Output path: {output_path}")
    print(f"Base name for inference: {base}")

    # Load the model and scaler
    print(f"Loading model from {model_file}...")
    model = tf.keras.models.load_model(model_file, custom_objects={'CombinedAUC': CombinedAUC})

    print(f"Loading scaler from {scaler_file}...")
    scaler = joblib.load(scaler_file)

    # Load the input data for inference
    print(f"Loading input data from {input_file}...")
    input_df = pd.read_parquet(input_file)
    print(f"Data loaded: {len(input_df)} rows, {len(input_df.columns)} columns")

    # Perform inference
    print("Making predictions...")
    output_df = make_inference(model, scaler, input_df, output_path, base)

    print("Inference completed.")


# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make inference using a trained model")
    parser.add_argument("--model_file", type=str, 
        help=f"Path to the trained model (default: {DEFAULT_CONFIG['model_file']})")
    parser.add_argument("--scaler_file", type=str, 
        help=f"Path to the saved scaler (default: {DEFAULT_CONFIG['scaler_file']})")
    parser.add_argument("--input_file", type=str, 
        help=f"Path to input data for inference (default: {DEFAULT_CONFIG['input_file']})")
    parser.add_argument("--output_path", type=str, 
        help=f"Path to save inference results (default: {DEFAULT_CONFIG['output_path']})")
    parser.add_argument("--rng", type=int, 
        help=f"Random seed (default: {DEFAULT_CONFIG['rng']})")
    args = parser.parse_args()

    # Merge config
    config = {**DEFAULT_CONFIG, **{k: v for k, v in vars(args).items() if v is not None}}
    
    main(config)
