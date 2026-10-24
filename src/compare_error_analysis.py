from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_text


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
FIGURE_DIR = Path("results/figures")
REPORT_DIR = Path("results/reports")

TEST_FILE = DATA_DIR / "test_preprocess.tsv"

BASELINE_MODEL_FILE = MODEL_DIR / "best_model.joblib"
BALANCED_MODEL_FILE = MODEL_DIR / "balanced_model.joblib"

LABELS = ["negative", "neutral", "positive"]

def load_test_data():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    df = pd.read_csv(
        TEST_FILE,
        sep="\t",
        header=None,
        names=["text", "label"],
    )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    df["processed_text"] = df["text"].apply(preprocess_text)
    df = df[df["processed_text"].str.strip() != ""].copy()

    invalid_labels = set(df["label"]) - set(LABELS)

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels found: {invalid_labels}"
        )

    return df.reset_index(drop=True)

def main():
    for directory in [FIGURE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=== Baseline vs Balanced Error Analysis ===")

    for model_file in [
        BASELINE_MODEL_FILE,
        BALANCED_MODEL_FILE,
    ]:
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}"
            )

    baseline_model = joblib.load(BASELINE_MODEL_FILE)
    balanced_model = joblib.load(BALANCED_MODEL_FILE)
    test_df = load_test_data()

    X_test = test_df["processed_text"]
    y_test = test_df["label"]

    print(f"\nTest samples: {len(test_df)}")
    print("Generating baseline predictions...")
    baseline_pred = baseline_model.predict(X_test)

    print("Generating balanced predictions...")
    balanced_pred = balanced_model.predict(X_test)

    comparison = pd.DataFrame({
        "text": test_df["text"],
        "actual": y_test,
        "baseline_predicted": baseline_pred,
        "balanced_predicted": balanced_pred,
    })

    comparison["baseline_correct"] = (
        comparison["actual"] == comparison["baseline_predicted"]
    )

    comparison["balanced_correct"] = (
        comparison["actual"] == comparison["balanced_predicted"]
    )

    def classify_transition(row):
        baseline_ok = row["baseline_correct"]
        balanced_ok = row["balanced_correct"]

        if baseline_ok and balanced_ok:
            return "both_correct"
        if not baseline_ok and balanced_ok:
            return "balanced_fixed_error"
        if baseline_ok and not balanced_ok:
            return "balanced_introduced_error"
        return "both_wrong"

    comparison["transition"] = comparison.apply(
        classify_transition,
        axis=1,
    )

    comparison_path = REPORT_DIR / "prediction_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    disagreement_df = comparison[
        comparison["baseline_predicted"]
        != comparison["balanced_predicted"]
    ].copy()

    disagreement_path = REPORT_DIR / "model_disagreements.csv"
    disagreement_df.to_csv(disagreement_path, index=False)

    both_wrong_df = comparison[
        comparison["transition"] == "both_wrong"
    ].copy()

    both_wrong_path = REPORT_DIR / "shared_errors.csv"
    both_wrong_df.to_csv(both_wrong_path, index=False)

    transition_order = [
        "both_correct",
        "balanced_fixed_error",
        "balanced_introduced_error",
        "both_wrong",
    ]

    transition_summary = (
        comparison["transition"]
        .value_counts()
        .reindex(transition_order, fill_value=0)
        .rename_axis("transition")
        .reset_index(name="samples")
    )

    transition_summary["percentage"] = (
        transition_summary["samples"] / len(comparison) * 100
    )

    transition_path = REPORT_DIR / "prediction_transition_summary.csv"
    transition_summary.to_csv(transition_path, index=False)

    baseline_cm = confusion_matrix(
        y_test,
        baseline_pred,
        labels=LABELS,
    )

    balanced_cm = confusion_matrix(
        y_test,
        balanced_pred,
        labels=LABELS,
    )

    # Save confusion matrices side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=baseline_cm,
        display_labels=LABELS,
    ).plot(
        ax=axes[0],
        values_format="d",
        colorbar=False,
    )
    axes[0].set_title("Baseline Linear SVM")

    ConfusionMatrixDisplay(
        confusion_matrix=balanced_cm,
        display_labels=LABELS,
    ).plot(
        ax=axes[1],
        values_format="d",
        colorbar=False,
    )
    axes[1].set_title("Balanced Linear SVM")

    fig.tight_layout()

    figure_path = FIGURE_DIR / "confusion_matrix_comparison.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Compare the most frequent error pairs.
    baseline_errors = comparison[
        ~comparison["baseline_correct"]
    ]

    balanced_errors = comparison[
        ~comparison["balanced_correct"]
    ]

    def error_pair_counts(df, prediction_column, model_name):
        counts = (
            df.groupby(["actual", prediction_column])
            .size()
            .reset_index(name="samples")
            .rename(columns={
                prediction_column: "predicted",
            })
        )

        counts["model"] = model_name
        counts = counts[counts["actual"] != counts["predicted"]]

        return counts[["model", "actual", "predicted", "samples"]]

    error_pairs = pd.concat([
        error_pair_counts(
            baseline_errors,
            "baseline_predicted",
            "baseline",
        ),
        error_pair_counts(
            balanced_errors,
            "balanced_predicted",
            "balanced",
        ),
    ], ignore_index=True)

    error_pairs = error_pairs.sort_values(
        ["model", "samples"],
        ascending=[True, False],
    )

    error_pairs_path = REPORT_DIR / "error_pair_comparison.csv"
    error_pairs.to_csv(error_pairs_path, index=False)

    print("\n=== Prediction Transition Summary ===")
    print(transition_summary.to_string(index=False))

    print("\n=== Misclassification Counts ===")
    print(f"Baseline errors : {len(baseline_errors)}")
    print(f"Balanced errors : {len(balanced_errors)}")
    print(f"Model disagreements: {len(disagreement_df)}")
    print(f"Both models wrong: {len(both_wrong_df)}")

    print("\n=== Error Pairs ===")
    print(error_pairs.to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {comparison_path}")
    print(f"- {disagreement_path}")
    print(f"- {both_wrong_path}")
    print(f"- {transition_path}")
    print(f"- {error_pairs_path}")
    print(f"- {figure_path}")

    print("\nError analysis comparison completed!")


if __name__ == "__main__":
    main()