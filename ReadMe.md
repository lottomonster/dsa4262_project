
---

# README

## Quick Start (Evaluate Repo using Test data)

> Follow these commands **in sequence**.
> Each line is designed to be copy-pasted individually.

> This project already includes a sample dataset in the repo which is located in `data/csv/test_dataset.csv`.

---
### 1. Install system dependencies (Ubuntu)

```bash
sudo apt update
```

```bash
sudo apt install python3 python3-pip git
```

---
### 2. Clone the repository

```bash
git clone https://github.com/lottomonster/dsa4262_project.git
```

```bash
cd dsa4262_project
```

---
### 3. Install required Python packages

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

---
### 4. Process the test dataset (provided in repo)

```bash
python3 scripts/data_processing.py --mode inference --input_file ../data/csv/test_dataset.csv
```

Expected output file:

```
/home/ubuntu/dsa4262_project/data/inference/test_dataset_inference_results.csv
```

---
### 5. Run inference using the trained model

```bash
python3 scripts/inference.py --input_file ../data/processed/test_dataset_processed.parquet
```

Expected output file:

```
/home/ubuntu/dsa4262_project/data/inference/test_dataset_inference_results.csv
```

<br>
<br>

## File Structure

```plaintext
dsa4262_project/
  ├── data/
  │   ├── csv/              # Store raw CSV files for data processing
  │   │   └── test_dataset.csv  # Prepared
  │   ├── processed/        # Store processed datasets (Parquet format) for training or inference
  │   │   ├── dataset0_processed.parquet  # Processed data for model training/inference
  │   │   ├── dataset1_processed.parquet  # Processed data for inference
  │   │   ├── dataset2_processed.parquet  # Processed data for inference
  │   │   └── dataset3_processed.parquet  # Processed data for inference
  │   ├── results/          # Results from the training process (models, metrics, logs)
  │   │   ├── dataset0_full_scaler.pkl  # Scaler generated from training.py
  │   │   └── dataset0_full_model.keras  # Model weights generated from running training.py
  │   ├── inference/        # Output folder for inference.py (submission 3 column format)
  │   │   ├── dataset0_inference.csv  # Model Prediction of dataset0
  │   │   ├── dataset1_inference.csv  # Model Prediction of dataset1
  │   │   ├── dataset2_inference.csv  # Model Prediction of dataset2
  │   │   └── dataset3_inference.csv  # Model Prediction of dataset3
  ├── scripts/
  │   ├── data_processing.py # Data preprocessing script
  │   ├── training.py        # Model training script
  │   └── inference.py       # Model inference script
  ├── ReadME.md
  └── requirements.txt
```
Run all cmds from dsa4262_project location thanks.


<br>
<br>

## Path Handling (Important)

For all scripts (`data_processing.py`, `training.py`, `inference.py`):

* Any **file or folder path** provided with flags (`--input_file`, `--output_path`, `--model_file`, `--scaler_file`) is **relative to where the script is stored**, not your current working directory (CWD).
* This ensures the script behaves the same no matter where you run it from.

---

### File/Folder Flags

| Flag            | Description                                                             | Default / Example Value                                                                            |
| --------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `--input_file`  | Input dataset (CSV for processing, Parquet for training/inference)      | `../data/csv/test_dataset.csv`                                                                     |
| `--output_path` | Folder where processed data, models, or inference results will be saved | `../data/processed/` (processing), `../data/results/` (training), `../data/inference/` (inference) |
| `--model_file`  | Trained model file for `inference.py`                                   | `../data/results/dataset0_full_model.keras`                                                        |
| `--scaler_file` | Saved scaler file for `inference.py`                                    | `../data/results/dataset0_full_scaler.pkl`                                                         |

---

### Example

Suppose `data_processing.py` is stored in `scripts/`:

```bash
python scripts/data_processing.py --input_file ../data/csv/test_dataset.csv
```

* Even if you run this command from **any folder**, the script will always use `../data/csv/test_dataset.csv` **relative to where data_processing.py is stored**, not your current working directory.


<br>
<br>

## End-to-End Pipeline (Architecture Overview)

```
                          ┌──────────────────────────┐
      Raw CSV Dataset ---►│   data_processing.py     │
 (test_dataset.csv,       │ (--mode train / infer)   │
  dataset0.csv)           └───────────┬──────────────┘
                                      │
                      ┌───────────────┴────────────────┐
                      │                                │
                      ▼                                ▼
          Processed dataset                Processed dataset
      for TRAINING (.parquet)           for INFERENCE (.parquet)                                  
      e.g. data/processed/              e.g. data/processed/
      dataset0_processed.parquet        test_dataset_processed.parquet
                      │                                │
                      ▼                                │
            ┌─────────────────┐                        │
            │   training.py   │                        │
            │  (train model)  │                        │
            └─────────┬───────┘                        │
                      │                                │
     saves output to: │                                │
                      ▼                                │
     ┌─────────────────────────────────────┐           │
     │     Model + Scaler Generated        │           │
     │   - dataset0_full_model.keras       │           │
     │   - dataset0_full_scaler.pkl        │           │
     │     stored in data/results/         │           │
     └───────────┬────────────────────────┘            │
                 │                                     │
                 └──────────────┬──────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      inference.py      │
                    │ (generate predictions) │
                    └───────────┬────────────┘
                                ▼
                    Predictions CSV generated
            e.g., data/inference/test_dataset_inference.csv


```
Note: The model and scaler generated by training.py are loaded by inference.py for prediction.


<br>
<br>
<br>

# Data Processing

The `data_processing.py` script processes datasets for model training and inference. It allows you to specify which dataset to use, the processing mode to prepare data for either training or inference, and more. Below are details on how to run the script and what flags are available.

## Running the Data Processing Script

By default, `data_processing.py` processes the data stored in `data/csv/test_dataset.csv` for inference.

### Default Command (Processing for Inference)
If no specific input file is provided, the script will default to processing `test_dataset.csv` from the `data/csv/` folder. The processed data will be saved as a .parquet file in the `data/processed/` folder. The output file will be prefixed with the base name of the input file.

```bash
python scripts/data_processing.py
```

### Specify a Different CSV Dataset

You can specify a different dataset for processing using the `--input_file` flag.


```bash
python scripts/data_processing.py --input_file ../data/csv/dataset1.csv
```

## Flags and Parameters

The script accepts the following command-line flags. Each flag configures a specific aspect of the data processing.

> **Note:** Both the `--input_file` and `--output_path` flags are **relative to the location of the `data_processing.py` script**, not the current working directory (CWD).

| Flag            | Description                                                                                                                    | Accepted Values                    | Default Value                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- | --------------------------------- |
| `--input_file`    | Path to the input CSV file containing your data. This is required to specify the dataset for processing. If not provided, the default file is `test_dataset.csv` | Path to a CSV file                 | `../data/csv/test_dataset.csv`       |
| `--output_path`   | Folder where the processed data will be saved.          | Path to a folder                | `../data/processed/`          |
| `--mode`         | The mode to run the script in. This determines the type of processing (training or inference).                                 | `train`, `inference`               | `inference` (default)                 |
| `--rng`          | A random seed to ensure reproducibility of any random operations during data processing. This is optional.                     | Any integer value (e.g. 42, 1234) | `42` | 


### Modes

#### **inference** (Default Mode)

When in **inference** mode, the script prepares data for inference (e.g. predictions or analysis without retraining the model). In this case, you **must** provide an input CSV file that does not require a `label` column.

Example command for inference mode:

```bash
python scripts/data_processing.py --mode inference --input_file ../data/csv/dataset0.csv
```

#### **train** 

When in **train** mode, the script processes the dataset for model training. The input dataset must contain the following columns:

* `gene_id`: Unique identifier for each gene.
* `label`: The target label for each row.

Example command for training mode:

```bash
python scripts/data_processing.py --mode train --input_file ../data/csv/dataset0.csv
```

### Argument Parsing with `--help`

If you're unsure about the flags or need a quick reference, you can always check the available arguments and their descriptions by running the script with the `--help` flag:

```bash
python scripts/data_processing.py --help
```

## When to Run the Data Processing Script

* **Run the data processing script in training mode** if you plan to train a model with the processed data.
* **Run the data processing script in inference mode** if you want to prepare data for inference or evaluation, without needing to train the model.

Make sure to use the appropriate dataset, input flags, and modes based on the task you are working on.



<br>
<br>
<br>

# Training

The `training.py` script is used to train a model based on a preprocessed parquet dataset. It allows you to configure various aspects of the training process, such as parquet dataset paths, random seed, train-test ratio, and more.

## Running the Training Script

By default, the `training.py` script uses the following configuration:

* **Input dataset**: `../data/processed/dataset0_processed.parquet`
* **Output folder**: `../data/results`
* **Random seed**: `42`
* **Validation metric**: Precision-Recall AUC for early stopping monitoring
* **K-fold cross-validation**: Disabled by default

### Default Command (Training the Model)

To run the training script with the default configuration, use the following command:

And if no specific input file is provided, the script will default to using `dataset0_processed.parquet` from the `data/processed/` folder. The results, model and scaler will be saved in the `data/results/` folder. The output files will be prefixed with the base name of the input file.

```bash
python scripts/training.py
```

### Specify a Different Parquet Dataset

You can specify a different parquet dataset for training using the `--input_file` flag.

> **Important:** The parquet dataset used for training must have been processed using `--mode train` in the `data_processing.py` script.


```bash
python scripts/training.py --input_file ../data/processed/dataset1_processed.parquet
```

## Flags and Parameters

The script accepts the following command-line flags. Each flag configures a specific aspect of the training process.

> **Note:** Both the `--input_file` and `--output_path` flags are **relative to where the `training.py` script is stored**, not the current working directory (CWD).

| Flag                 | Description                                                                                                                 | Accepted Values                    | Default Value                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------- |
| `--input_file`      | Path to the input dataset in Parquet format. This is the file containing the processed dataset for training.                | Path to a Parquet file             | `../data/processed/dataset0_processed.parquet` |
| `--output_path`     | Folder where the training results will be saved (e.g., model checkpoints logs).                                         | Path to a folder                | `../data/results`                              |
| `--rng`             | Random seed to ensure reproducibility of training.                                                                          | Any integer value (e.g. 42, 1234) | `42`                                           |
| `--split_ratio`     | Train/Test split ratio. Provide two float values that sum to 1.0 (e.g. `0.8 0.2` for 80% training and 20% testing).        | Two float values that sum to 1.0   | `0.85 0.15` (85% Training, 15% Testing)            |
| `--monitor`         | Validation metric to monitor for early stopping. Options: `val_pr_auc` or `val_roc_auc`.                                    | `val_pr_auc`, `val_roc_auc`        | `val_pr_auc`                                   |
| `--patience`        | The number of epochs to wait for improvement in the validation metric before early stopping is triggered.                   | Any integer value                  | `20`                                           |
| `--run_kfold`       | Enable or disable k-fold cross-validation. You can specify either `True` or `False`  (case-insensitive).                     | `True`, `False`                    | `False`                                        |
| `--k_fold_splits`   | The number of splits for k-fold cross-validation. Must be ≥ 2 to enable k-fold validation.                                  | Any integer ≥ 2                    | `5`                                            |
| `--run_full_model`  | Train the final model on the full dataset (no test split). This is useful for final production model training. You can specify either `True` or `False`  (case-insensitive).              | `True`, `False`                    | `True`                                         |
| `--train_val_ratio` | Train/validation split ratio used when training the full model. Provide two float values that sum to 1.0 (e.g. `0.9 0.1`). | Two float values that sum to 1.0   | `0.95 0.05` (95% Training, 5% Validation)               |

### Example Usage:

1. **Train the model with the default settings** (input file `dataset0_processed.parquet`, output path `../data/results`):

   ```bash
   python scripts/training.py
   ```

2. **Specify a differnt dataset for training** (e.g., `dataset1_processed.parquet`):

   ```bash
   python scripts/training.py --input_file ../data/processed/dataset1_processed.parquet
   ```

3. **Specify a different folder to store results, trained models and scalar** :

   ```bash
   python scripts/training.py --output_path ../new_results_folder
   ```

4. **Enable k-fold cross-validation with 3 splits**:

   ```bash
   python scripts/training.py --run_kfold True --k_fold_splits 3
   ```

5. **Use ROC AUC for validation metric (e.g., `val_roc_auc`)**:

   ```bash
   python scripts/training.py --monitor val_roc_auc
   ```

6. **Train the model on the full dataset (no test split)**:

   ```bash
   python scripts/training.py --run_full_model True
   ```

7. **Set a custom train/test split ratio** (e.g., 70% training and 30% testing):

   ```bash
   python scripts/training.py --split_ratio 0.7 0.3
   ```

8. **Set a custom train/validation split ratio for training a model on entire dataset** (e.g., 98% training and 2% validation):

   ```bash
   python scripts/training.py --train_val_ratio 0.98 0.02
   ```

9. **Set a custom random seed**:

   ```bash
   python scripts/training.py --rng 1234
   ```

10. **Set a custom patience for early stopping** (e.g. stop after 30 epochs of no improvement):

   ```bash
   python scripts/training.py --patience 30
   ```

### Argument Parsing with `--help`

If you're unsure about the flags or need a quick reference, you can always check the available arguments and their descriptions by running the script with the `--help` flag:

```bash
python scripts/training.py --help
```


## When to Run the Training Script

* **Run the training script** after the data has been processed using `data_processing.py` with the `--mode train` option.
* **Ensure the processed dataset is in Parquet format** and ready for training before executing this script.

<br>
<br>
<br>

# Inference

The `inference.py` script is used to apply a trained model to new data. It generates prediction probabilities (scores) using an existing model and scaler. This script is primarily for making predictions on unseen data after the model has been trained.

## Running the Inference Script

By default, `inference.py` uses the data stored in `data/processed/test_dataset_processed.parquet` for inference.

### Default Command (Inference)

If no specific input file is provided, the script will default to using `test_dataset_processed.parquet` in the `data/processed/` folder. The results, model and scaler will be saved in the `data/inference/` folder. The output files will be prefixed with the base name of the input file.

```bash
python scripts/training.py
```

### Specify a Different Input File or Model

You can specify different files for inference (e.g., a new input dataset or different trained model) using the following flags:

```bash
python scripts/inference.py --input_file ../data/processed/dataset1_processed.parquet --model_file ../data/results/dataset0_full_model.keras --scaler_file ../data/results/dataset0_full_scaler.pkl
```

## Flags and Parameters

The script accepts the following command-line flags. Each flag is used to configure the script's behavior. **Note** that both `--input_file`, `--output_path`, `--model_file`, and `--scaler_file` are **relative to where the `inference.py` script is stored**, not the current working directory (CWD).

| Flag            | Description                                                                                                                                         | Accepted Values               | Default Value                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------- |
| `--model_file`  | Path to the trained model (e.g., a Keras model).                                                                                                    | Path to a model file (.keras) | `../data/results/dataset0_full_model.keras`        |
| `--scaler_file` | Path to the saved scaler (e.g., a `.pkl` file used for scaling the features).                                                                       | Path to a scaler file (.pkl)  | `../data/results/dataset0_full_scaler.pkl`         |
| `--input_file`  | Path to the input dataset in Parquet format. This is the file containing the processed dataset for inference.                                                      | Path to a `.parquet` file     | `../data/processed/test_dataset_processed.parquet` |
| `--output_path` | Folder where the inference results will be saved (CSV file).                                                                                     | Path to a folder           | `../data/inference`                                |
| `--rng`         | Random seed to ensure reproducibility of any random operations.                                                                                     | Any integer (e.g., 42, 1234)  | `42`                                               |
| `--full_keep`   | If `True`, the script will also return a full dataset (with a new column for `score`). If `False`, only the columns needed for submission will be saved. | `True`, `False`               | `False`                                            |

### Example Usage:

1. **Run inference with the default settings** (input file `test_dataset_processed.parquet`, model `dataset0_full_model.keras`, scaler `dataset0_full_scaler.pkl`):

   ```bash
   python scripts/inference.py
   ```

2. **Specify a different input file for inference** (e.g., `dataset1_processed.parquet`):

   ```bash
   python scripts/inference.py --input_file ../data/processed/dataset1_processed.parquet
   ```

3. **Specify a different model and scaler for inference**:

   ```bash
   python scripts/inference.py --model_file ../data/results/datasetX_full_model.keras --scaler_file ../data/results/datasetX_full_scaler.pkl
   ```

4. **Change the output folder for inference results**:

   ```bash
   python scripts/inference.py --output_path ../data/new_inference_folder
   ```

5. **Set a custom random seed for reproducibility**:

   ```bash
   python scripts/inference.py --rng 1234
   ```

6. **Save the full dataset along with prediction scores** (instead of just the necessary columns for submission):

   ```bash
   python scripts/inference.py --full_keep True
   ```

7. **Run with a different model file and input file**:

   ```bash
   python scripts/inference.py --model_file ../data/results/model_v2.keras --input_file ../data/processed/new_test_dataset.parquet
   ```

### Argument Parsing with `--help`

If you're unsure about the flags or need a quick reference, you can always check the available arguments and their descriptions by running the script with the `--help` flag:

```bash
python scripts/inference.py --help
```


## How Inference Works

1. **Input Data**: The script expects a processed dataset in `.parquet` format, containing columns such as `transcript_id`, `transcript_position`, and any relevant features for making predictions.

2. **Scaling**: The input data is passed through the pre-trained scaler to standardize the feature values. This ensures the input data matches the scale of the data the model was trained on.

3. **Model Prediction**: The script loads the pre-trained model (e.g., a Keras model) and uses it to generate continuous probabilities (scores) between 0 and 1 for each row in the input data.

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
<br>
<br>


## Final Notes:

1. **File Paths**: Always ensure that the `input_file` and `output_path` are provided relative to the `scripts` folder. For example, when running the script from the root directory, you'll need to reference files from the `../data/csv/` and `../data/processed/` paths.

2. **File Size**: Large files (e.g., `dataset0_processed.parquet`) may not be included in the GitHub repository. Therefore, good luck have fun figure out yourself.

---