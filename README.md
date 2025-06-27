# ML Training: Fraud Detection

This repository contains resources and code to explore, train, and evaluate machine learning models for fraud detection. We leverage both synthetic mobile-money transaction data (PaySim) and real credit card transaction data to build robust classifiers.

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Getting Started](#getting-started)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Logging](#logging)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

Fraud detection is critical for financial systems. In this project, we:

* Conduct initial data exploration with Jupyter notebooks.
* Preprocess and clean transaction datasets.
* Train several machine learning models (e.g., XGBoost, Random Forest).
* Evaluate performance using precision, recall, ROC-AUC, and more.

Our goal is to provide a clear, reproducible pipeline for training and testing fraud detection models.

---

## Dataset

Raw and processed data are stored under version control with Git LFS for large files.

* **Raw datasets (LFS-tracked)**

  * `dataset/creditcard.csv` — Real-world credit card transactions (<100 MB chunks).
  * `dataset/PaySim_Synthetic_Mobile-Money_Simulator_dataset.csv` — Synthetic mobile-money transactions.

* **Additional data directories**

  * `data/raw/` — For any extra raw inputs or logs.
  * `data/processed/` — (Optional) Processed or sampled versions of the raw data.

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* Git with Git LFS installed and enabled

### Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:Julyjujuy/ML_train_fraud_detection.git
   cd ML_train_fraud_detection
   ```
2. Set up a virtual environment:

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```plaintext
ML_train_fraud_detection/
├── 01_explore_data.ipynb    # Quick exploration notebook
├── data/                   # Raw and processed data folders
│   ├── raw/                # Raw input files
│   └── processed/          # Cleaned or sampled data (optional)
├── dataset/                # LFS-tracked large CSV datasets
├── notebooks/              # Experiment notebooks
├── src/                    # Training and inference scripts
├── tests/                  # Unit and integration tests
├── models/                 # Saved model artifacts
├── github/                 # GitHub config (actions, templates)
├── requirements.txt        # Python dependencies
├── shadow_inference.log    # Inference log file
└── README.md               # This file
```

---

## Usage

### Data Exploration

Open `01_explore_data.ipynb` or any notebook in `notebooks/` to explore dataset statistics and visualize distributions.

### Model Training

Run the training script in `src/`:

```bash
python src/train.py --config configs/train_config.yaml
```

### Inference

Perform inference on new data:

```bash
python src/inference.py --input data/raw/new_transactions.csv --output results/predictions.csv
```

---

## Evaluation

Evaluate model performance with:

```bash
python src/evaluate.py --predictions results/predictions.csv --labels dataset/creditcard.csv
```

Metrics reported include precision, recall, F1-score, and ROC-AUC.

---

## Logging

All inference runs are logged to `shadow_inference.log`. You can review it for timestamps, input sizes, and aggregate metrics.

---

## Contributing

We welcome improvements! Please:

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes and push: `git push origin feature/your-feature`.
4. Open a pull request and describe your work.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
