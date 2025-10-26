
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
  │   └── outputs/          # Outputs folder for inference.py (submission 3 column format)
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

### Running the Script

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

To run inference, use the `inference.py` script. This script is still a work in progress (WIP), so additional documentation will be provided once it's finalized.

### Running Inference

```bash
python scripts/inference.py
```

---

### Final Notes:

1. **File Paths**: Always remember that the `input_file` and `output_path` paths are relative to the `scripts` folder. For example, if you're running the script from the root directory, you'll need to reference files from the `../data/csv/` path.
2. **File Size**: Some data files like `dataset0.csv` may be too large to include in the GitHub repo. Therefore, good luck have fun figure out yourself.

---
