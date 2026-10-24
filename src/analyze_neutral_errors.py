from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import classification_report
from preprocessing import preprocess_text


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
REPORT_DIR = Path("results/reports")

TEST_FILE = DATA_DIR / "test_preprocess.tsv"
BASELINE_MODEL_FILE = MODEL_DIR / "best_model.joblib"
BALANCED_MODEL_FILE = MODEL_DIR / "balanced_model.joblib"

LABELS = ["negative", "neutral", "positive"]


def load_test_data():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {TEST_FILE}")

    df = pd.read_csv(
        TEST_FILE,
        sep="\t",
        header=None,
        names=["text", "label"],
    )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    invalid_labels = set(df["label"]) - set(LABELS)
    if invalid_labels:
        raise ValueError(f"Unexpected labels: {invalid_labels}")

    df["processed_text"] = df["text"].apply(preprocess_text)
    df = df[df["processed_text"].str.strip() != ""].copy()

    return df.reset_index(drop=True)

def get_decision_scores(model, texts):
    """
    Return LinearSVC decision scores where available.
    These scores are not calibrated probabilities.
    """
    classifier = model.named_steps["classifier"]
    vectorizer = model.named_steps["tfidf"]

    transformed = vectorizer.transform(texts)
    scores = classifier.decision_function(transformed)

    if scores.ndim == 1:
        return pd.DataFrame({"decision_score": scores})

    return pd.DataFrame(
        scores,
        columns=classifier.classes_,
    )

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Neutral Sentiment Error Analysis ===")

    for model_path in [BASELINE_MODEL_FILE, BALANCED_MODEL_FILE]:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    test_df = load_test_data()

    baseline_model = joblib.load(BASELINE_MODEL_FILE)
    balanced_model = joblib.load(BALANCED_MODEL_FILE)

    X_test = test_df["processed_text"]
    y_test = test_df["label"]

    baseline_predictions = baseline_model.predict(X_test)
    balanced_predictions = balanced_model.predict(X_test)

    results = test_df[["text", "label"]].rename(
        columns={"label": "actual"}
    ).copy()

    results["baseline_predicted"] = baseline_predictions
    results["balanced_predicted"] = balanced_predictions

    results["baseline_correct"] = (
        results["actual"] == results["baseline_predicted"]
    )
    results["balanced_correct"] = (
        results["actual"] == results["balanced_predicted"]
    )

    neutral_df = results[results["actual"] == "neutral"].copy()

    neutral_df["error_pattern"] = neutral_df.apply(
        lambda row: (
            "both_correct"
            if row["baseline_correct"] and row["balanced_correct"]
            else "balanced_fixed_error"
            if not row["baseline_correct"] and row["balanced_correct"]
            else "balanced_introduced_error"
            if row["baseline_correct"] and not row["balanced_correct"]
            else "both_wrong"
        ),
        axis=1,
    )

    neutral_errors = neutral_df[
        ~neutral_df["balanced_correct"]
    ].copy()

    neutral_all_path = REPORT_DIR / "neutral_test_predictions.csv"
    neutral_errors_path = REPORT_DIR / "neutral_errors_balanced.csv"

    neutral_df.to_csv(neutral_all_path, index=False)
    neutral_errors.to_csv(neutral_errors_path, index=False)

    scores = get_decision_scores(balanced_model, X_test)
    scores.index = results.index

    score_columns = [
        label for label in LABELS if label in scores.columns
    ]

    if score_columns:
        neutral_errors = neutral_errors.join(scores[score_columns])
        neutral_errors["predicted_class_score"] = neutral_errors.apply(
            lambda row: row.get(row["balanced_predicted"], float("nan")),
            axis=1,
        )

        neutral_errors.to_csv(neutral_errors_path, index=False)

    prediction_counts = (
        neutral_df["balanced_predicted"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .rename_axis("predicted_label")
        .reset_index(name="samples")
    )

    prediction_counts["percentage_of_neutral"] = (
        prediction_counts["samples"] / len(neutral_df) * 100
    )

    counts_path = REPORT_DIR / "neutral_prediction_counts.csv"
    prediction_counts.to_csv(counts_path, index=False)

    # Print reports.
    print(f"\nTotal test samples: {len(results)}")
    print(f"Actual neutral samples: {len(neutral_df)}")
    print(f"Balanced neutral errors: {len(neutral_errors)}")

    print("\n=== Balanced Predictions for Actual Neutral Samples ===")
    print(prediction_counts.to_string(index=False))

    print("\n=== Neutral Error Patterns ===")
    print(
        neutral_df["error_pattern"]
        .value_counts()
        .reindex(
            [
                "both_correct",
                "balanced_fixed_error",
                "balanced_introduced_error",
                "both_wrong",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\n=== Classification Report (Balanced Model) ===")
    print(
        classification_report(
            y_test,
            balanced_predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("\n=== Examples of Neutral Errors ===")
    if neutral_errors.empty:
        print("No neutral errors found.")
    else:
        columns_to_show = [
            "text",
            "baseline_predicted",
            "balanced_predicted",
        ]
        print(
            neutral_errors[columns_to_show]
            .head(20)
            .to_string(index=False)
        )

    print("\nSaved reports:")
    print(f"- {neutral_all_path}")
    print(f"- {neutral_errors_path}")
    print(f"- {counts_path}")

    print("\nNeutral error analysis completed!")


if __name__ == "__main__":
    main()