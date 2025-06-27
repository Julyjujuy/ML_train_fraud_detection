import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load

from src.features import basic_clean_cc


def setup_logging(log_file: str):
    """
    Configure logging to output INFO-level messages to both a file and the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def run_shadow_inference(prod_model_path: Path, new_model_path: Path, input_df: pd.DataFrame):
    """
    Run inference on both production and new models, log per-transaction results and a summary.
    """
    # Load models
    prod_model = load(prod_model_path)
    new_model = load(new_model_path)

    # Apply feature pipeline
    df_clean = basic_clean_cc(input_df)
    X = df_clean.drop('Class', axis=1, errors='ignore')

    # Predict probabilities and classes
    prod_probs = prod_model.predict_proba(X)[:, 1]
    prod_preds = prod_model.predict(X)
    new_probs = new_model.predict_proba(X)[:, 1]
    new_preds = new_model.predict(X)

    # Log each transaction
    for idx, (pp, py, np_, ny) in enumerate(zip(prod_probs, prod_preds, new_probs, new_preds)):
        logging.info(
            f"tx_index={idx} prod_pred={py} prod_prob={pp:.4f} "
            f"new_pred={ny} new_prob={np_:.4f}"
        )

    # Summary statistics
    total = len(prod_preds)
    disagrees = int((prod_preds != new_preds).sum())
    new_only = int(((prod_preds == 0) & (new_preds == 1)).sum())
    prod_only = int(((prod_preds == 1) & (new_preds == 0)).sum())

    logging.info(
        f"SUMMARY: total={total}, disagrees={disagrees}, "
        f"new_only={new_only}, prod_only={prod_only}"
    )

    return prod_preds, new_preds


def main():
    parser = argparse.ArgumentParser(
        description="Run shadow-mode inference comparing production and new fraud models"
    )
    parser.add_argument(
        "--prod_model", required=True,
        help="Path to the production model joblib file"
    )
    parser.add_argument(
        "--new_model", required=True,
        help="Path to the new (trained) model joblib file"
    )
    parser.add_argument(
        "--input_csv", required=True,
        help="Path to CSV file containing transactions to score"
    )
    parser.add_argument(
        "--log_file", default="shadow_inference.log",
        help="File to write inference logs to"
    )
    args = parser.parse_args()

    setup_logging(args.log_file)
    logging.info("Starting shadow inference")

    # Load input transactions
    df_input = pd.read_csv(args.input_csv)
    run_shadow_inference(
        Path(args.prod_model), Path(args.new_model), df_input
    )
    logging.info("Shadow inference completed")


if __name__ == '__main__':
    main()
