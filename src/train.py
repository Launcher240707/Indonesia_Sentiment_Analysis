from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocessing import preprocess_text

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
REPORT_DIR = Path("results/reports")

TRAIN_FILE = DATA_DIR / "train_preprocess.tsv"
VALID_FILE = DATA_DIR / "valid_preprocess.tsv"

RANDOM_STATE = 42

def load_dataset(file_path):
    """Load a labeled sentiment dataset from a TSV file."""
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

def prepare_text(df):
    """Apply text preprocessing to a dataset."""
    df = df.copy()

    df["text"] = df["text"].apply(preprocess_text)

    # Remove empty text after preprocessing.
    df = df[df["text"].str.strip() != ""].copy()

    return df

def build_model(classifier):
    """Build a TF-IDF and classifier pipeline."""
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
        ("classifier", classifier),
    ])

def evaluate_model(name, model, X_valid, y_valid):
    """Evaluate a model using the validation dataset."""
    predictions = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    macro_f1 = f1_score(
        y_valid,
        predictions,
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
            labels=["negative", "neutral", "positive"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_valid,
            predictions,
            labels=["negative", "neutral", "positive"],
        )
    )

    return {
        "model_name": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

def main():
    print("=== Indonesian Sentiment Analysis ===")
    print("Stage: Model Training and Validation")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    valid_df = load_dataset(VALID_FILE)

    print(f"\nOriginal training samples   : {len(train_df)}")
    print(f"Original validation samples: {len(valid_df)}")

    validation_texts = set(valid_df["text"])

    overlap_mask = train_df["text"].isin(validation_texts)
    overlap_count = int(overlap_mask.sum())

    train_df = train_df.loc[~overlap_mask].copy()

    print(f"Overlapping training rows removed: {overlap_count}")
    print(f"Training samples after cleanup   : {len(train_df)}")

    train_df = prepare_text(train_df)
    valid_df = prepare_text(valid_df)

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_valid = valid_df["text"]
    y_valid = valid_df["label"]

    print(f"\nTraining samples after preprocessing   : {len(X_train)}")
    print(f"Validation samples after preprocessing: {len(X_valid)}")

    models = {
        "Logistic Regression": build_model(
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        ),
        "Linear SVM": build_model(
            LinearSVC(
                random_state=RANDOM_STATE,
            )
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            model,
            X_valid,
            y_valid,
        )

        results.append(metrics)
        trained_models[name] = model

    # Compare models using validation Macro-F1.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="macro_f1",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save comparison report.
    report_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(report_path, index=False)

    best_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_name]

    best_model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    print(f"\nBest model: {best_name}")
    print(f"Saved model: {best_model_path}")
    print(f"Saved comparison report: {report_path}")

    print("\nModel training and validation completed!")


if __name__ == "__main__":
    main()
