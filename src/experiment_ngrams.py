from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC
from preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_preprocess.tsv"
VALID_PATH = ROOT / "data" / "raw" / "valid_preprocess.tsv"
REPORT_DIR = ROOT / "results" / "reports"

LABELS = ["negative", "neutral", "positive"]
NGRAM_CONFIGS = [(1, 1), (1, 2)]


def load_dataset(path):
    """Load the Indonesian sentiment dataset."""
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_PATH)
    valid_df = load_dataset(VALID_PATH)

    validation_texts = set(valid_df["text"])
    overlap_mask = train_df["text"].isin(validation_texts)
    overlap_count = int(overlap_mask.sum())

    train_df = train_df.loc[~overlap_mask].copy()

    print("=== TF-IDF N-gram Experiment ===")
    print(f"Training samples after overlap removal: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Removed overlapping training rows: {overlap_count}")

    X_train = train_df["text"].map(preprocess_text)
    X_valid = valid_df["text"].map(preprocess_text)

    y_train = train_df["label"]
    y_valid = valid_df["label"]

    results = []
    confusion_records = []

    for ngram_range in NGRAM_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Testing ngram_range={ngram_range}")
        print("=" * 60)

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_valid_tfidf = vectorizer.transform(X_valid)

        model = LinearSVC(
            class_weight="balanced",
            random_state=42,
        )

        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_valid_tfidf)

        accuracy = accuracy_score(y_valid, predictions)
        macro_f1 = f1_score(
            y_valid,
            predictions,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            y_valid,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        neutral_recall = report["neutral"]["recall"]
        neutral_f1 = report["neutral"]["f1-score"]
        neutral_precision = report["neutral"]["precision"]

        results.append(
            {
                "ngram_range": str(ngram_range),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "neutral_precision": neutral_precision,
                "neutral_recall": neutral_recall,
                "neutral_f1": neutral_f1,
                "vocabulary_size": len(vectorizer.vocabulary_),
            }
        )

        matrix = confusion_matrix(
            y_valid,
            predictions,
            labels=LABELS,
        )

        for i, actual_label in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_records.append(
                    {
                        "ngram_range": str(ngram_range),
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "count": int(matrix[i, j]),
                    }
                )

        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Neutral precision: {neutral_precision:.4f}")
        print(f"Neutral recall: {neutral_recall:.4f}")
        print(f"Neutral F1: {neutral_f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_valid,
                predictions,
                labels=LABELS,
                zero_division=0,
            )
        )

        print("Confusion Matrix (actual rows, predicted columns):")
        print(pd.DataFrame(
            matrix,
            index=LABELS,
            columns=LABELS,
        ))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_path = REPORT_DIR / "ngram_experiment.csv"
    confusion_path = REPORT_DIR / "ngram_confusion_matrices.csv"

    results_df.to_csv(results_path, index=False)
    pd.DataFrame(confusion_records).to_csv(
        confusion_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("=== FINAL N-GRAM COMPARISON ===")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest configuration by validation Macro-F1: "
          f"{best['ngram_range']}")
    print(f"Macro-F1: {best['macro_f1']:.4f}")
    print(f"Neutral recall: {best['neutral_recall']:.4f}")
    print(f"Neutral F1: {best['neutral_f1']:.4f}")

    print(f"\nSaved report: {results_path.relative_to(ROOT)}")
    print(f"Saved confusion matrices: {confusion_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()