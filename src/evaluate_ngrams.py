from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "train_preprocess.tsv"
VALID_PATH = ROOT / "data" / "raw" / "valid_preprocess.tsv"
TEST_PATH = ROOT / "data" / "raw" / "test_preprocess.tsv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "results" / "reports"
FIGURE_DIR = ROOT / "results" / "figures"

MODEL_PATH = MODEL_DIR / "ngram_balanced_model.joblib"
METRICS_PATH = REPORT_DIR / "ngram_test_metrics.csv"
ERROR_PATH = REPORT_DIR / "ngram_test_errors.csv"
PREDICTIONS_PATH = REPORT_DIR / "ngram_test_predictions.csv"
CONFUSION_PATH = FIGURE_DIR / "ngram_confusion_matrix.png"

LABELS = ["negative", "neutral", "positive"]


def load_dataset(path):
    """Load a labeled Indonesian sentiment dataset."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        keep_default_na=False,
    )

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    return df


def main():
    for directory in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_PATH)
    valid_df = load_dataset(VALID_PATH)
    test_df = load_dataset(TEST_PATH)

    validation_texts = set(valid_df["text"])
    overlap_mask = train_df["text"].isin(validation_texts)
    removed_rows = int(overlap_mask.sum())

    train_df = train_df.loc[~overlap_mask].copy()

    print("=== Final TF-IDF N-gram Test Evaluation ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Removed overlapping training rows: {removed_rows}")
    print("Configuration: LinearSVC + balanced class weights")
    print("TF-IDF ngram_range=(1, 2)")

    X_train = train_df["text"].map(preprocess_text)
    y_train = train_df["label"]

    X_test = test_df["text"].map(preprocess_text)
    y_test = test_df["label"]

    model = Pipeline(
        [
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
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    print("\n=== Test Results ===")
    print(f"Test samples: {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "LinearSVC balanced TF-IDF (1,2)",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "test_samples": len(test_df),
        "removed_training_overlap_rows": removed_rows,
        "vocabulary_size": len(
            model.named_steps["tfidf"].vocabulary_
        ),
    }

    for label in LABELS:
        metrics[f"{label}_precision"] = report[label]["precision"]
        metrics[f"{label}_recall"] = report[label]["recall"]
        metrics[f"{label}_f1"] = report[label]["f1-score"]

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    predictions_df = pd.DataFrame(
        {
            "text": test_df["text"],
            "actual": y_test,
            "predicted": predictions,
        }
    )
    predictions_df["correct"] = (
        predictions_df["actual"] == predictions_df["predicted"]
    )
    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    errors_df = predictions_df.loc[
        ~predictions_df["correct"]
    ].copy()
    errors_df.to_csv(
        ERROR_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Save confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("TF-IDF Unigram + Bigram - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=300)
    plt.close()

    print("\n=== Saved Artifacts ===")
    for path in [
        MODEL_PATH,
        METRICS_PATH,
        ERROR_PATH,
        PREDICTIONS_PATH,
        CONFUSION_PATH,
    ]:
        print(path.relative_to(ROOT))

    print(f"\nTotal test errors: {len(errors_df)}")
    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()