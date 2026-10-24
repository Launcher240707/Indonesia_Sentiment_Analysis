from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
PUBLIC_DIR = REPORTS_DIR / "public"

FILES_TO_CLEAN = [
    "error_analysis.csv",
    "error_analysis_balanced.csv",
    "model_disagreements.csv",
    "neutral_errors_balanced.csv",
    "neutral_error_feature_contributions.csv",
    "neutral_test_predictions.csv",
    "ngram_prediction_comparison.csv",
    "ngram_test_errors.csv",
    "ngram_test_predictions.csv",
    "prediction_comparison.csv",
    "shared_errors.csv",
]


def main():
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES_TO_CLEAN:
        source = REPORTS_DIR / filename

        if not source.exists():
            print(f"SKIPPED (not found): {filename}")
            continue

        df = pd.read_csv(source)

        if "text" in df.columns:
            df = df.drop(columns=["text"])

        destination = PUBLIC_DIR / filename
        df.to_csv(destination, index=False)

        print(
            f"Saved: {destination.relative_to(PROJECT_ROOT)} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )

    print("\nPublic report preparation complete.")


if __name__ == "__main__":
    main()
