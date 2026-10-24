from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocessing import preprocess_text


DATA_DIR = Path("data/raw")
REPORT_DIR = Path("results/reports")

TRAIN_FILE = DATA_DIR / "train_preprocess.tsv"
VALID_FILE = DATA_DIR / "valid_preprocess.tsv"

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42

def load_dataset(file_path):
    """Load a labeled TSV dataset."""
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
    """Preprocess text and remove empty samples."""
    df = df.copy()
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df["text"].str.strip() != ""].copy()

    return df

def build_model(class_weight=None):
    """Build a Linear SVM pipeline."""
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
                class_weight=class_weight,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

def evaluate_model(name, model, X_valid, y_valid):
    """Evaluate a model on the validation set."""
    predictions = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(
        y_valid,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_valid,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

def main():
    print("=== Experiment: Linear SVM Class Weighting ===")

    train_df = load_dataset(TRAIN_FILE)
    valid_df = load_dataset(VALID_FILE)

    # Prevent exact raw-text overlap between train and validation.
    validation_texts = set(valid_df["text"])
    overlap_mask = train_df["text"].isin(validation_texts)

    print(f"\nOriginal training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Overlapping training rows removed: {overlap_mask.sum()}")

    train_df = train_df.loc[~overlap_mask].copy()

    train_df = prepare_dataset(train_df)
    valid_df = prepare_dataset(valid_df)

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_valid = valid_df["text"]
    y_valid = valid_df["label"]

    experiments = {
        "Linear SVM - Baseline": None,
        "Linear SVM - Balanced": "balanced",
    }

    results = []

    for name, class_weight in experiments.items():
        print(f"\nTraining {name}...")

        model = build_model(class_weight=class_weight)
        model.fit(X_train, y_train)

        result = evaluate_model(
            name,
            model,
            X_valid,
            y_valid,
        )
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "class_weight_experiment.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nComparison saved to: {output_path}")
    print("Experiment completed!")


if __name__ == "__main__":
    main()