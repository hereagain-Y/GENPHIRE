"""
Generalized phenotype sentence builder.
Usage:
    python build_sentences.py --input toy_data.csv --output toy_sentences.csv

You can also customize:
    --id_col ID
    --top_n 20
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Build phenotype sentences from a dataset.")

    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", required=True, help="Where to save the final sentence CSV.")
    parser.add_argument("--id_col", default="ID", help="ID column (default: ID).")
    
    parser.add_argument("--top_n", type=int, default=20, help="Top N traits to select (default: 25).")
    parser.add_argument(
        "--trait_col",
        type=int,
        default=None,
        help="1-based index of the first trait column (e.g., 9 means start from the 9th column)."
    )


    return parser.parse_args()


def default_clinical_covariates():
    """Default covariates expected in the toy dataset."""
    return {
        "male": lambda x: "Male" if x == 1 else "Female",
        "age_enrolled": lambda x: f"{int(x)} years old" if not pd.isna(x) else "Age unknown",
        "BMI": lambda x: f"BMI {x:.1f}" if not pd.isna(x) else "BMI unknown",
        "smoking": lambda x: "smoker" if x in [1,2] else "non-smoker"
    }


def build_sentences(input_csv, output_csv, id_col="ID",top_n=20,trait_col_start=9):
    print(f"📂 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    clinical_covs = default_clinical_covariates()
    exclude = set([id_col] + list(clinical_covs.keys()))

    if trait_col_start is not None:
        # user-specified start column (1-based → convert to 0-based index)
        start_idx = trait_col_start - 1
        trait_cols = list(df.columns[start_idx:])
    else:
    # auto-detect numeric trait columns excluding ID + clinical covariates
        trait_cols = [
        c for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]

    print(f"🧬 Found {len(trait_cols)} trait columns starting at: {trait_cols[0]}")
    # === Normalize traits ===
    scaler = StandardScaler()
    df[trait_cols] = scaler.fit_transform(df[trait_cols])

    # === Get top-N traits ===
    trait_matrix = df[trait_cols].values
    top_idx = np.argsort(-trait_matrix, axis=1)[:, :top_n]
    top_trait_names = [
        ", ".join([trait_cols[i] for i in row])
        for row in top_idx
    ]

    # === Build clinical summary strings ===
    def clinical_string(row):
        parts = [fn(row[col]) for col, fn in clinical_covs.items()]
        return ", ".join(parts)

    df["sentence"] = df.apply(clinical_string, axis=1) + ", " + pd.Series(top_trait_names)

    # === Save final output ===
    df_out = df[[id_col, "sentence"]]
    df_out.to_csv(output_csv, index=False)

    print(f"🎉 Saved final phenotype sentences to: {output_csv}")


if __name__ == "__main__":
    args = parse_args()
    build_sentences(
        input_csv=args.input,
        output_csv=args.output,
        id_col=args.id_col,
        top_n=args.top_n,
        trait_col_start=args.trait_col
    )
