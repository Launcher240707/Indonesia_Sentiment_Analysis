from pathlib import Path

import joblib

from preprocessing import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "balanced_model.joblib"


def load_model():
    """Load the selected final sentiment classification model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run the model training/evaluation script first."
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(model, text: str) -> str:
    """Predict the sentiment of a single Indonesian text."""
    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        raise ValueError("Input text must not be empty.")

    prediction = model.predict([cleaned_text])[0]
    return str(prediction)


def main():
    model = load_model()

    print("Indonesian Sentiment Analysis")
    print("Final model: Balanced Linear SVM + TF-IDF (unigram + bigram)")
    print("Enter Indonesian text to predict its sentiment.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            text = input("Text: ").strip()

            if text.lower() == "exit":
                print("Program closed.")
                break

            if not text:
                print("Please enter some text.\n")
                continue

            sentiment = predict_sentiment(model, text)
            print(f"Predicted sentiment: {sentiment.upper()}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nProgram closed.")
            break


if __name__ == "__main__":
    main()
