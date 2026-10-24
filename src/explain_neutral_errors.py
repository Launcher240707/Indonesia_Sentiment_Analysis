from pathlib import Path

import joblib
import pandas as pd

from preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "balanced_model.joblib"
ERROR_PATH = ROOT / "results" / "reports" / "neutral_errors_balanced.csv"
OUTPUT_PATH = (
    ROOT
    / "results"
    / "reports"
    / "neutral_error_feature_contributions.csv"
)


def top_feature_contributions(vectorizer, classifier, text, label, top_n=5):
    """Find the strongest positive TF-IDF feature contributions."""
    cleaned_text = preprocess_text(text)
    vector = vectorizer.transform([cleaned_text])

    class_index = list(classifier.classes_).index(label)
    contributions = vector.multiply(
        classifier.coef_[class_index]
    ).toarray().ravel()

    feature_names = vectorizer.get_feature_names_out()

    positive_indices = [
        i for i, value in enumerate(contributions) if value > 0
    ]
    positive_indices.sort(
        key=lambda i: contributions[i],
        reverse=True,
    )

    return "; ".join(
        f"{feature_names[i]} ({contributions[i]:.4f})"
        for i in positive_indices[:top_n]
    )


def main():
    model = joblib.load(MODEL_PATH)
    errors = pd.read_csv(ERROR_PATH)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    records = []

    for _, row in errors.iterrows():
        text = row["text"]
        predicted = row["balanced_predicted"]

        records.append(
            {
                "text": text,
                "actual": row["actual"],
                "predicted": predicted,
                "predicted_score": row["predicted_class_score"],
                "neutral_score": row["neutral"],
                "top_predicted_features": top_feature_contributions(
                    vectorizer, classifier, text, predicted
                ),
                "top_neutral_features": top_feature_contributions(
                    vectorizer, classifier, text, "neutral"
                ),
            }
        )

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== Neutral Error Feature Contributions ===")
    print(f"Analyzed errors: {len(result)}")
    print(f"Saved report: {OUTPUT_PATH.relative_to(ROOT)}")

    columns = [
        "actual",
        "predicted",
        "predicted_score",
        "neutral_score",
        "top_predicted_features",
        "top_neutral_features",
    ]

    print("\n=== First 10 Examples ===")
    print(result[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()