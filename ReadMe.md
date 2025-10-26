
---

# Project README

## File Structure

```plaintext
root/
  ├── data/
  │   ├── json/             # Store JSON files (SGNEX data or other data in JSON format)
  │   ├── csv/              # Store CSV files (tabular data for data_processing.py)
  │   │   ├── dataset0.csv  # Large file (not in GitHub)
  │   │   ├── dataset1.csv  # Large file (not in GitHub)
  │   │   └── dataset2.csv  # Added CSV file
  │   ├── processed/        # Store processed CSV files for training or inference
  │   │   ├── dataset0_processed.parquet  # Processed data for model training/inference
  │   │   └── dataset1_processed.parquet  # Processed data for model inference
  │   ├── results/          # Results folder for training.py (models, metrics, png)
  │   └── inference/        # Output folder for inference.py (submission 3 column format)
  └── scripts/
      ├── __init__.py        # (Legacy, may be removed)
      ├── sh/                # Additional scripts (e.g., "cherron" for custom functions)
      ├── data_processing.py # Data preprocessing script
      ├── training.py        # Model training script
      └── inference.py       # Model inference script
```
Run all cmds from root location thanks bai.
## Data Processing

By default, `data_processing.py` processes the data from `data/csv/dataset0.csv` for model training. To specify another dataset, you must pass the `--input_file` flag with the relative path to your desired CSV file.

### Running the Data Processing Script

To run the default data processing (using `dataset0.csv`):

```bash
python scripts/data_processing.py
```

To specify a different dataset (e.g., `dataset1.csv` for inference mode):

```bash
python scripts/data_processing.py --mode inference --input_file ../data/csv/dataset1.csv
```

### Modes

* **train** (default): The dataset must contain the `gene_id` and `label` columns for training.
* **inference**: Use this mode to prepare data for inference. In this case, the `input_file` flag is required to specify the dataset.

### Example Code:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing step 1")
    parser.add_argument("--mode", type=str, choices=["train", "inference"],
                        help=f"Run mode (default: {DEFAULT_CONFIG['mode']})")
    parser.add_argument("--input_file", type=str,
                        help=f"Path to input CSV (default: {DEFAULT_CONFIG['input_file']})")
    parser.add_argument("--output_path", type=str,
                        help=f"Output directory (default: {DEFAULT_CONFIG['output_path']})")
    parser.add_argument("--rng", type=int,
                        help=f"Random seed (default: {DEFAULT_CONFIG['rng']})")
```

## Training

The training script should only be run after the data has been processed using `data_processing.py` with the `--mode train` option.

### Running the Training Script

```bash
python scripts/training.py
```

You can specify a different dataset for training using the `--input_file` flag:

```bash
python scripts/training.py --input_file ../data/processed/dataset0_processed.parquet
```

### Arguments

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or inference a model")
    parser.add_argument("--input_file", type=str, help="Path to input parquet file")
    parser.add_argument("--output_path", type=str, help="Output directory")
    parser.add_argument("--rng", type=int, help="Random seed")
    parser.add_argument("--split_ratios", type=str, help="Train/Test split as JSON")
    parser.add_argument("--monitor", type=str, choices = ['val_pr_auc', 'val_roc_auc', 'val_roc_plus_pr'], help="Validation Metric used for early stopping")
    parser.add_argument("--patience", type=int, help="Epochs with no improvement for early stopping")
    parser.add_argument("--k_fold_splits", type=int, help="Number of k-fold splits")
```

## Inference

To run inference on a pre-trained model, use the `inference.py` script. This script allows you to apply a trained model to new data, scale it with an existing scaler, and generate prediction probabilities (scores).

### Running the Inference Script

```bash
python scripts/inference.py
```

You can specify a different input file for inference using the `--input_file` flag or choose different models to infer on using `--model_file` and `--scaler_file` flags:

```bash
python scripts/inference.py --input_file ../data/processed/dataset1_processed.parquet --model_file ../data/results/dataset0_full_model.keras --scaler_file ../data/results/dataset0_full_scaler.pkl
```

### Arguments

```python
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
```

### How Inference Works

1. **Input Data**: The script expects a processed dataset (`.parquet` file) containing columns such as `transcript_id`, `transcript_position`, and any relevant features for making predictions.

2. **Scaling**: The input data is passed through the pre-trained scaler to standardize the feature values. This step is necessary to ensure the input data matches the scale of the data the model was trained on.

3. **Model Prediction**: The script loads a pre-trained model (e.g., a Keras model) and uses it to generate continuous probabilities between 0 and 1 for each row in the input data.

4. **Output**: The script creates a new dataframe containing the following columns:

   * `transcript_id`
   * `transcript_position`
   * `score` (the predicted probability from the model)

   This dataframe is saved as a CSV file in the specified `output_path`.

### Example Output

The resulting CSV will have the following structure:

| transcript_id | transcript_position | score |
| ------------- | ------------------- | ----- |
| 101           | 1                   | 0.785 |
| 102           | 1                   | 0.912 |
| 103           | 2                   | 0.423 |
| 104           | 2                   | 0.658 |
| ...           | ...                 | ...   |

Where `score` is the predicted probability for each row.

---

### Final Notes:

1. **File Paths**: Always ensure that the `input_file` and `output_path` are provided relative to the `scripts` directory. For example, when running the script from the root directory, you'll need to reference files from the `../data/csv/` and `../data/processed/` paths.

2. **File Size**: Large files (e.g., `dataset0_processed.parquet`) may not be included in the GitHub repository. Therefore, good luck have fun figure out yourself.

---