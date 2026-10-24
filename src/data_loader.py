from pathlib import Path

import pandas as pd
import requests

BASE_URL = (
    "https://raw.githubusercontent.com/IndoNLP/indonlu/master/"
    "dataset/smsa_doc-sentiment-prosa"
)

FILES = {
    "train": "train_preprocess.tsv",
    "validation": "valid_preprocess.tsv",
    "test": "test_preprocess.tsv",
}

OUTPUT_DIR = Path("data/raw")

def download_file(filename: str) -> Path:
    """Download a dataset file if it does not already exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    if output_path.exists():
        print(f"[SKIP] {filename} already exists.")
        return output_path
    url = f"{BASE_URL}/{filename}"
    print(f"[DOWNLOAD] {filename}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"[SAVED] {output_path}")
    return output_path

def inspect_dataset(name: str, path: Path) -> None:
    """Display basic information about a dataset."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
    )
    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 50}")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())
    print("\nMissing values:")
    print(df.isnull().sum())

def main():
    print("=== Indonesian Sentiment Analysis Dataset Loader ===\n")
    downloaded_files = {}
    for name, filename in FILES.items():
        downloaded_files[name] = download_file(filename)
    print("\n=== Dataset Inspection ===")
    for name, path in downloaded_files.items():
        inspect_dataset(name, path)
    print("\nDataset loading and inspection completed.")

if __name__ == "__main__":
    main()