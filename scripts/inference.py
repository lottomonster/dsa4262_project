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

    # Make predictions
    y_pred_prob = model.predict(X_input_scaled)

    # Prepare the 3 columns required for submission
    output_df = input_df[["transcript_id", "transcript_position"]].copy()
    output_df["score"] = y_pred_prob  # Inference probability as the score
    output_file = output_path / f"{base}_inference_results.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")
    
    # Merge with full dataset if full_keep_df is True
    if full_keep_df:
        full_output_df = input_df.copy()
        full_output_df["score"] = y_pred_prob
        full_output_df = full_output_df
        full_output_file = output_path / f"{base}_full_inference_results.csv"
        full_output_df.to_csv(full_output_file, index=False)
        print(f"\nInference results (FULL) saved to {full_output_file}")

    # Save the results to a CSV file
    

    return output_df

# Very impt messages, do not delete
messages = [
    "Inference completed! Give us 5 stars if this ran faster than your last-minute assignment scramble 📝💨",
    "Inference completed! Faster than your brain processing caffeine at 8 AM ☕⚡",
    "Inference completed! Quicker than your WiFi when everyone’s streaming the season finale 📶😎",
    "Inference completed! Like a group project saving grace, but solo 🤓",
    "Inference completed! Rapid like your panic when the professor drops a surprise quiz 😱",
    "Inference completed! Like teleportation for your data—blink and it’s done! 🛸",
    "Inference completed! Like instant noodles for data—ready in minutes 🍜",
    "Inference completed! Give it an A+ if it ran faster than your last all-nighter 🚀📚",
    "Inference completed! Rate it 10/10 if this script finished before you finished your coffee ☕💨",
    "Inference completed! Give it full marks if your WiFi survived finals week better than this ran 📶😎",
    "Inference completed! Award it a perfect score if it ran quicker than you submitted assignments last minute ⏱️💯",
    "Inference completed! Give it a 5-star if it’s as smooth as that one group project that actually went well ⭐⭐⭐⭐⭐",
    "Inference completed! Rate it top-tier if this script finished faster than your panic during exam prep 😅📖",
    "Inference completed! Give it an A if it was more reliable than your roommate showing up to class 📚🕒",
    "Inference completed! Full marks if it ran faster than you hitting ‘Submit’ before the deadline 🏃‍♂️💨",
    "Inference completed! Rate it 100% if it finished quicker than your favorite streaming episode binge 📺⚡",
    "Inference completed! Give it a perfect score if it ran smoother than your morning coffee brewing ☕✨",
    "Inference completed! Award it 5 stars if it’s faster than your friend Cherron running these scripts again 🖥️💨",
    "Inference completed! Full marks if this script beat your record for waiting for results 📊🚀",
    "Inference completed! Rate it top score if it ran faster than your group chat exploded with memes 📱🤣",
    "Inference completed! Give it A+ if it finished before you started procrastinating 🤓⌛",
    "Inference completed! Perfect rating if it’s smoother than your last essay submission 📝✨",
    "Inference completed! Full marks if this ran quicker than you avoided Zoom calls this week 😎💻",
    "Inference completed! Award 5 stars if it’s faster than me trying to debug these scripts ☁️💨",
    "Inference completed! Rate it top-notch if it’s quicker than your snack break between classes 🍿⚡",
    "Inference completed! Give it full points if it finished faster than your last group assignment argument 😂📝",
    "Inference completed! 5 stars if it ran more efficiently than your email inbox cleanup 📧⚡",
    "Inference completed! Perfect score if it ran faster than your last lecture nap 😴🚀",
    "Inference completed! Full marks if it finished quicker than Cherron re-running the inference scripts again 🖥️💨",
    "Inference completed! Give it a top rating if it was faster than your campus coffee line ☕🏃",
    "Inference completed! Rate it 100% if it ran faster than your last panic-search for lecture notes 📓💨",
    "Inference completed! Perfect score if it ran smoother than your thesis formatting ✨📄",
    "Inference completed! Full marks if it’s quicker than your group meeting to assign chores 😂👨‍💻",
    "Inference completed! Give it 5 stars if it’s faster than your last last-minute print job 🖨️💨",
    "Inference completed! Top score if it ran quicker than your brain during finals week 🧠⚡",
    "Inference completed! Award A+ if it’s faster than your last TikTok scroll break 📱💨",
    "Inference completed! Give full points if it ran smoother than your first draft actually working ✍️✨",
    "Inference completed! 5-star if it finished before your friend Cherron remembered he had to run it again 🖥️😂",
    "Inference completed! Fast as your roommate finishing the last slice of pizza 🍕",
    "Inference completed! Like your favorite app opening before you can blink 📱",
    "Inference completed! Quick like your group project meeting ending early 🏁",
    "Inference completed! Rapid like your caffeine-fueled coding session ☕💻",
    "Inference completed! Speedy like your panic realizing finals are next week 😱",
    "Inference completed! Like your GPA after an extra credit miracle, but faster 📈✨",
    "Inference completed! Rapid like your coffee disappearing during a coding marathon ☕💻",
    "Inference completed! Faster than your group chat deciding on lunch 🍔",
    "Inference completed! Quicker than your brain during finals week panic mode 😅",
    "Inference completed! Like Cherron refreshing the cloud terminal for the tenth time 🖥️💨",
    "Inference completed! Speedy like your roommate finishing all the snacks 🍫",
    "Inference completed! Fast as your favorite meme going viral 📲",
    "Inference completed! Like Cherron staring at logs wondering why the script crashed again 👀",
    "Inference completed! Quicker than your Spotify playlist skipping ads 🎵",
    "Inference completed! Rapid like a last-minute group project miracle 💾",
    "Inference completed! Fast like your caffeine-fueled midnight coding session ☕🌙",
    "Inference completed! Like teleportation for your dataset, but without the mess 🛸",
    "Inference completed! Speedy like Cherron having to re-run the cloud job three times 😅",
    "Inference completed! Quick like your brain processing 8 AM lecture notes 🧠",
    "Inference completed! Like instant noodles for your data—ready in minutes 🍜",
    "Inference completed! Faster than your panic realizing deadlines are today ⏰",
    "Inference completed! Rapid like your group project last-minute save 🏁",
    "Inference completed! Like Cherron juggling multiple cloud runs at once ☁️💻",
    "Inference completed! Quick like your favorite app opening before you can blink 📱",
    "Inference completed! Speedy like your panic when the professor announces extra credit 😱"
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
    print("\n" + random.choice(messages))


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
