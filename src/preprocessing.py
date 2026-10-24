import re
import unicodedata


def preprocess_text(text: str) -> str:
    """Apply minimal preprocessing to Indonesian text."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Convert text to lowercase
    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " email ",
        text,
    )

    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    examples = [
        "MAKANANNYA enak banget!!!",
        "Aku tidak suka pelayanan ini :(",
        "Cek https://example.com untuk info.",
        "Email kami di info@example.com",
    ]

    for example in examples:
        print("Original :", example)
        print("Processed:", preprocess_text(example))
        print("-" * 50)
