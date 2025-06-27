import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def basic_clean_cc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the Credit Card dataset:
    - Scales the 'Amount' column to zero mean and unit variance.
    - Drops the original 'Amount'.
    """
    df = df.copy()
    scaler = StandardScaler()
    # Fit-transform expects a 2D array
    df["ScaledAmount"] = scaler.fit_transform(df[["Amount"]])
    df.drop("Amount", axis=1, inplace=True)
    return df


def basic_clean_ps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the PaySim dataset:
    - One-hot encodes the 'type' categorical feature.
    - Log-transforms highly skewed monetary and balance fields.
    - Drops the original raw columns after transformation.
    """
    df = df.copy()
    # One-hot encode transaction type, dropping the first level to avoid collinearity
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Log-transform skewed monetary and balance fields
    monetary_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    for col in monetary_cols:
        df[f"log_{col}"] = np.log1p(df[col])  # add 1 to avoid log(0)

    # Drop raw columns
    df.drop(monetary_cols, axis=1, inplace=True)
    return df
