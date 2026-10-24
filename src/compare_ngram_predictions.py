from pathlib import Path

import joblib
import pandas as pd

from preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test_preprocess.tsv"

MODELS = {
    "balanced": ROOT / "models" / "balanced_model.joblib",
    "ngram_balanced": ROOT / "models" / "ngram_balanced_model.joblib",
}

OUTPUT_DIR = ROOT / "results" / "reports"


def main():
    test_df = pd.read_csv(
        TEST_PATH,
        sep="\t",
        header=None,
        names=["text", "label"],
        keep_default_na=False,
    )

    texts = test_df["text"].astype(str)
    actual = test_df["label"].astype(str).str.strip().str.lower()
    cleaned_texts = texts.map(preprocess_text)

    results = pd.DataFrame({
        "text": texts,
        "actual": actual,
    })

    for name, model_path in MODELS.items():
        model = joblib.load(model_path)
        results[f"{name}_predicted"] = model.predict(cleaned_texts)
        results[f"{name}_correct"] = (
            results[f"{name}_predicted"] == actual
        )

    results["predictions_agree"] = (
        results["balanced_predicted"]
        == results["ngram_balanced_predicted"]
    )

    results["error_pattern"] = "both_correct"

    both_wrong = (
        ~results["balanced_correct"]
        & ~results["ngram_balanced_correct"]
    )
    balanced_fixed = (
        ~results["balanced_correct"]
        & results["ngram_balanced_correct"]
    )
    ngram_fixed = (
        results["balanced_correct"]
        & ~results["ngram_balanced_correct"]
    )

    results.loc[both_wrong, "error_pattern"] = "both_wrong"
    results.loc[balanced_fixed, "error_pattern"] = "ngram_fixed_error"
    results.loc[ngram_fixed, "error_pattern"] = "ngram_introduced_error"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "ngram_prediction_comparison.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=== Balanced vs N-gram Prediction Comparison ===")
    print(f"Test samples: {len(results)}")
    print(
        "Identical predictions:",
        int(results["predictions_agree"].sum()),
    )
    print(
        "Different predictions:",
        int((~results["predictions_agree"]).sum()),
    )

    print("\n=== Error Transition Summary ===")
    print(results["error_pattern"].value_counts().to_string())

    print("\n=== Prediction Agreement by Actual Class ===")
    print(
        results.groupby("actual")["predictions_agree"]
        .agg(["sum", "count", "mean"])
        .to_string()
    )

    print(f"\nSaved report: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()