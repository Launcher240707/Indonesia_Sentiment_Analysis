from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from preprocessing import preprocess_text


TEST_FILE = Path("data/raw/test_preprocess.tsv")
MODEL_FILE = Path("models/best_model.joblib")

FIGURE_DIR = Path("results/figures")
REPORT_DIR = Path("results/reports")

LABELS = ["negative", "neutral", "positive"]

def load_test_dataset(file_path):
    """Load the labeled test dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["text", "label"],
    )

    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    return df

def main():
    print("=== Indonesian Sentiment Analysis ===")
    print("Stage: Final Test Evaluation")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}. "
            "Run python src/train.py first."
        )

    model = joblib.load(MODEL_FILE)

    test_df = load_test_dataset(TEST_FILE)

    print(f"\nTest samples: {len(test_df)}")
    print("\nTest label distribution:")
    print(test_df["label"].value_counts().reindex(LABELS, fill_value=0))

    X_test = test_df["text"].apply(preprocess_text)
    y_test = test_df["label"]

    valid_mask = X_test.str.strip() != ""
    X_test = X_test.loc[valid_mask]
    y_test = y_test.loc[valid_mask]
    test_texts = test_df.loc[valid_mask, "text"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS,
    )

    display.plot(values_format="d")
    plt.title("Indonesian Sentiment Analysis - Test Set")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    error_df = pd.DataFrame({
        "text": test_texts.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    })

    error_df = error_df[
        error_df["actual"] != error_df["predicted"]
    ].copy()

    error_path = REPORT_DIR / "error_analysis.csv"
    error_df.to_csv(error_path, index=False)

    metrics_df = pd.DataFrame([{
        "model": "Linear SVM",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "test_samples": len(y_test),
        "misclassified_samples": len(error_df),
    }])

    metrics_path = REPORT_DIR / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nSaved outputs:")
    print(f"Confusion matrix : {figure_path}")
    print(f"Error analysis   : {error_path}")
    print(f"Test metrics     : {metrics_path}")

    print(f"\nMisclassified samples: {len(error_df)}")
    print("\nFinal evaluation completed!")


if __name__ == "__main__":
    main()