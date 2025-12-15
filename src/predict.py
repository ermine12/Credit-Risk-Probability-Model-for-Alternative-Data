from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd



def predict(model_path: str, input_csv: str, output_csv: str) -> None:
    model = joblib.load(model_path)

    df = pd.read_csv(input_csv)

    # The pipeline handles all preprocessing, so we pass the raw dataframe
    proba = model.predict_proba(df)[:, 1]

    out = df.copy()
    out["prob_default"] = proba

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/model.joblib")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="data/processed/predictions.csv")
    args = parser.parse_args()

    predict(args.model_path, args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
