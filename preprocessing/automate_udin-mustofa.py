import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def automate_preprocessing_pipeline(train_path, test_path, save_dir):
    # === Load data ===
    data = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop kolom tidak dipakai
    data = data.drop(columns="Tempat lahir", axis=1)
    test = test.drop(columns="Tempat lahir", axis=1)

    # Definisikan fitur
    numeric_features = data.select_dtypes(include=["number"]).columns
    categorical_features = data.select_dtypes(exclude=["number"]).columns

    log_features = ["Trigliserida (mg/dL)", "Glukosa Puasa (mg/dL)"]
    cat_features = ["Jenis Kelamin"]
    target = "Cholesterol Total (mg/dL)"

    # === Transformer ===
    log_transformer = Pipeline(steps=[
        ("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)),
        ("scaler", RobustScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first"))
    ])

    def binning_usia(X):
        bins = [18, 24, 29, 34, 39]
        labels = [0, 1, 2, 3]
        return pd.cut(
            X.squeeze(), bins=bins, labels=labels, include_lowest=True
        ).astype(int).to_frame()

    usia_transformer = Pipeline(steps=[
        ("binning", FunctionTransformer(binning_usia, validate=False))
    ])

    num_transformer = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    other_features = [
        col for col in numeric_features
        if col not in log_features and col not in [target, "Usia"]
    ]

    clean_dataframe = lambda df: df.dropna().drop_duplicates()
    clean_transformer = FunctionTransformer(clean_dataframe, validate=False)

    # === Pipeline utama ===
    pipe = Pipeline(steps=[
        ("clean", clean_transformer),
        ("preprocess", ColumnTransformer(
            transformers=[
                ("log_num", log_transformer, log_features),
                ("num", num_transformer, other_features),
                ("cat", cat_transformer, cat_features),
                ("enco", usia_transformer, ["Usia"])
            ]
        ))
    ])

    all_features = [
        *log_features,
        *other_features,
        *cat_features,
        "Usia"
    ]

    # === Split fitur & target ===
    X_train = data.drop(target, axis=1)
    y_train = data[target]

    X_test = test.drop(target, axis=1)
    y_test = test[target]

    # === Fit hanya di train ===
    pipe.fit(X_train)

    # === Transform train & test ===
    X_train_trans = pipe.transform(X_train)
    X_test_trans = pipe.transform(X_test)

    X_train_trans = pd.DataFrame(X_train_trans, index=X_train.index, columns=all_features)
    X_test_trans = pd.DataFrame(X_test_trans, index=X_test.index, columns=all_features)

    train_processed = X_train_trans.copy()
    train_processed[target] = y_train

    test_processed = X_test_trans.copy()
    test_processed[target] = y_test

    # === Simpan ke CSV ===
    os.makedirs(save_dir, exist_ok=True)
    train_processed.to_csv(os.path.join(save_dir, "train_processed.csv"), index=False)
    test_processed.to_csv(os.path.join(save_dir, "test_processed.csv"), index=False)

    print(f"[INFO] Saved train & test preprocessed data to {save_dir}")

# === CLI entrypoint ===
if __name__ == "__main__":
    train_path = sys.argv[1]   # path train.csv
    test_path = sys.argv[2]    # path test.csv
    save_dir = sys.argv[3]     # folder output

    automate_preprocessing_pipeline(train_path, test_path, save_dir)
