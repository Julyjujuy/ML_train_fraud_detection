import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from joblib import dump

from src.data_ingest import load_dataframe
from src.features import basic_clean_cc, basic_clean_ps


def main():
    # --- CreditCard Dataset Pipeline ---
    # 1. Load raw data
    df_cc = load_dataframe("creditcard.csv")
    # 2. Clean and feature-engineer
    df_cc = basic_clean_cc(df_cc)

    # 3. Split into features/labels
    X = df_cc.drop("Class", axis=1)
    y = df_cc["Class"]

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Handle class imbalance via under-sampling
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)

    # 6. Train the model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        n_estimators=100,
    )
    model.fit(X_res, y_res)

    # 7. Save the model
    dump(model, "models/creditcard_xgb.joblib")
    print("Saved CreditCard model to models/creditcard_xgb.joblib")

    # --- PaySim Dataset Pipeline (optional) ---
    # To train on PaySim you can uncomment and adapt below:
    # df_ps = load_dataframe("PaySim_Synthetic_Mobile-Money-Simulator_dataset.csv")
    # df_ps = basic_clean_ps(df_ps)
    # X_ps = df_ps.drop(["isFraud","isFlaggedFraud"], axis=1)
    # y_ps = df_ps["isFraud"]
    # # ... repeat split, resample, train steps ...


if __name__ == "__main__":
    main()
