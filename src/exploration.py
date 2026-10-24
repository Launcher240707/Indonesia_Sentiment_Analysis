from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results/figures")

DATASETS = {
    "Train": "train_preprocess.tsv",
    "Validation": "valid_preprocess.tsv",
    "Test": "test_preprocess.tsv",
}

def load_dataset(filename):
    """Load a sentiment dataset from a TSV file."""
    file_path = DATA_DIR / filename

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

    return df

def inspect_dataset(name, df):
    """Display basic statistics for a dataset."""
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

    duplicate_count = df["text"].duplicated().sum()
    print(f"\nDuplicate texts: {duplicate_count}")

    average_length = df["text"].str.len().mean()
    print(f"Average text length: {average_length:.2f} characters")

def inspect_duplicates(name, df):
    """Inspect duplicated texts and conflicting labels."""
    print(f"\n=== Duplicate Inspection: {name} ===")

    duplicated = df[
        df.duplicated(subset="text", keep=False)
    ]

    if duplicated.empty:
        print("No duplicated texts found.")
        return

    label_counts = duplicated.groupby("text")["label"].nunique()

    conflicting_texts = label_counts[
        label_counts > 1
    ]

    print(f"Rows involved in duplicates: {len(duplicated)}")
    print(f"Duplicated text groups: {len(label_counts)}")
    print(
        "Texts with conflicting labels: "
        f"{len(conflicting_texts)}"
    )

    if not conflicting_texts.empty:
        print("\nExamples of conflicting labels:")

        examples = duplicated[
            duplicated["text"].isin(conflicting_texts.index)
        ]

        print(examples.head(10).to_string(index=False))

def inspect_cross_split_duplicates(datasets):
    """Check whether texts overlap across dataset splits."""
    print(f"\n{'=' * 50}")
    print("CROSS-SPLIT DUPLICATE INSPECTION")
    print(f"{'=' * 50}")

    train_texts = set(datasets["Train"]["text"].dropna())
    valid_texts = set(datasets["Validation"]["text"].dropna())
    test_texts = set(datasets["Test"]["text"].dropna())

    print(
        "Train ∩ Validation:",
        len(train_texts & valid_texts),
    )

    print(
        "Train ∩ Test:",
        len(train_texts & test_texts),
    )

    print(
        "Validation ∩ Test:",
        len(valid_texts & test_texts),
    )

def save_label_distribution(datasets):
    """Create and save a label distribution chart."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label_counts = pd.DataFrame({
        name: df["label"].value_counts()
        for name, df in datasets.items()
    }).fillna(0)

    label_counts = label_counts.reindex(
        ["negative", "neutral", "positive"]
    )

    ax = label_counts.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title("Sentiment Label Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Samples")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    output_path = RESULTS_DIR / "label_distribution.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nChart saved to: {output_path}")

def inspect_cross_split_examples(train_df, valid_df):
    """Display texts shared by training and validation."""
    print("\n=== Shared Texts: Train and Validation ===")

    shared_texts = (
        set(train_df["text"].dropna())
        & set(valid_df["text"].dropna())
    )

    if not shared_texts:
        print("No shared texts found.")
        return

    train_shared = train_df[
        train_df["text"].isin(shared_texts)
    ][["text", "label"]].rename(
        columns={"label": "train_label"}
    )

    valid_shared = valid_df[
        valid_df["text"].isin(shared_texts)
    ][["text", "label"]].rename(
        columns={"label": "validation_label"}
    )

    comparison = train_shared.merge(
        valid_shared,
        on="text",
        how="inner",
    ).drop_duplicates()

    print(f"Shared unique texts: {len(shared_texts)}")
    print(comparison.to_string(index=False))

def main():
    print("=== Indonesian Sentiment Analysis: Exploration ===")

    datasets = {}

    for name, filename in DATASETS.items():
        df = load_dataset(filename)
        datasets[name] = df

        inspect_dataset(name, df)

    for name, df in datasets.items():
        inspect_duplicates(name, df)

    inspect_cross_split_duplicates(datasets)

    save_label_distribution(datasets)


    inspect_cross_split_duplicates(datasets)

    inspect_cross_split_examples(
        datasets["Train"],
        datasets["Validation"],
    )

    save_label_distribution(datasets)

    print("\nExploration completed successfully.")

if __name__ == "__main__":
    main()