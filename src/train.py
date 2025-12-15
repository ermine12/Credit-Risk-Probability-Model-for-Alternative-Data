from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import make_preprocess_pipeline


def train(input_csv: str, target_column: str, model_out: str, use_woe: bool = False) -> None:
    df = pd.read_csv(input_csv)

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    preprocess_pipe = make_preprocess_pipeline(use_woe=use_woe)

    model = Pipeline(
        steps=[
            ("preprocess", preprocess_pipe),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--use-woe", action="store_true")
    args = parser.parse_args()

    train(args.input_csv, args.target_column, args.model_out, args.use_woe)


if __name__ == "__main__":
    main()
