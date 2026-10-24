from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocessing import preprocess_text


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
FIGURE_DIR = Path("results/figures")
REPORT_DIR = Path("results/reports")

TRAIN_FILE = DATA_DIR / "train_preprocess.tsv"
VALID_FILE = DATA_DIR / "valid_preprocess.tsv"
TEST_FILE = DATA_DIR / "test_preprocess.tsv"

MODEL_FILE = MODEL_DIR / "balanced_model.joblib"

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42

def load_dataset(file_path):
    """Load a labeled dataset from a TSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
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

def prepare_dataset(df):
    """Apply the project's text preprocessing."""
    df = df.copy()
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df["text"].str.strip() != ""].copy()

    return df

def build_model():
    """Build the balanced Linear SVM pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

def main():
    print("=== Indonesian Sentiment Analysis ===")
    print("Stage: Balanced Linear SVM Test Evaluation")

    for directory in [MODEL_DIR, FIGURE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    valid_df = load_dataset(VALID_FILE)

    print(f"\nOriginal training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")

    validation_texts = set(valid_df["text"])
    overlap_mask = train_df["text"].isin(validation_texts)

    print(f"Overlapping training rows removed: {overlap_mask.sum()}")

    train_df = train_df.loc[~overlap_mask].copy()

    test_df = load_dataset(TEST_FILE)

    print(f"Test samples: {len(test_df)}")

    train_df = prepare_dataset(train_df)
    test_df = prepare_dataset(test_df)

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    print("\nTraining balanced Linear SVM...")
    model = build_model()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print(f"Saved balanced model: {MODEL_FILE}")

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
    print("BALANCED LINEAR SVM - FINAL TEST RESULTS")
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
    plt.title("Balanced Linear SVM - Test Set")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "confusion_matrix_balanced.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    error_df = pd.DataFrame({
        "text": test_df["text"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    })

    error_df = error_df[
        error_df["actual"] != error_df["predicted"]
    ]

    error_path = REPORT_DIR / "error_analysis_balanced.csv"
    error_df.to_csv(error_path, index=False)

    metrics_df = pd.DataFrame([{
        "model": "Linear SVM Balanced",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "test_samples": len(y_test),
        "misclassified_samples": len(error_df),
    }])

    metrics_path = REPORT_DIR / "test_metrics_balanced.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nSaved outputs:")
    print(f"Confusion matrix : {figure_path}")
    print(f"Error analysis   : {error_path}")
    print(f"Test metrics     : {metrics_path}")
    print(f"Misclassified samples: {len(error_df)}")

    print("\nBalanced test evaluation completed!")


if __name__ == "__main__":
    main()