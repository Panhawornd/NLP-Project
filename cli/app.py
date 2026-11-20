from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Project.cli.preprocessing import (
    average_word2vec,
    identity_tokenizer,  # Imported so joblib can resolve the saved vectorizer tokenizer.
    tokenize_normalize,
)

MODEL_PATH = Path(__file__).resolve().parent / "emotion_model.joblib"


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise SystemExit(
            "Trained model not found. Run TED-NLP.ipynb through its final refit/export cell first."
        )
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as exc:
        raise SystemExit(f"Could not load the saved model: {exc}") from exc


def predict_emotions(text: str, bundle: dict) -> list[tuple[str, float]]:
    try:
        kind = bundle["kind"]
        model = bundle["model"]
        if kind in {"pipeline", "sparse", "word2vec"}:
            tokens = tokenize_normalize(
                text,
                remove_stopwords=bundle["remove_stopwords"],
                lemmatize=bundle["lemmatize"],
            )
            if kind == "pipeline":
                probabilities = model.predict_proba([tokens])[0]
            elif kind == "sparse":
                features = bundle["vectorizer"].transform([tokens])
                probabilities = model.predict_proba(features)[0]
            else:
                features = average_word2vec([tokens], bundle["embedding"])
                probabilities = model.predict_proba(features)[0]
        else:
            raise ValueError(f"Unknown model type: {kind}")
    except LookupError as exc:
        raise SystemExit(
            "NLTK data is missing. Run the NLTK setup cell in TED-NLP.ipynb first."
        ) from exc

    labels = bundle["emotion_labels"]
    classes = bundle["classes"]
    results = [
        (labels[int(class_id)], float(probability) * 100.0)
        for class_id, probability in zip(classes, probabilities)
    ]
    return sorted(results, key=lambda item: item[1], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict anger, joy, optimism, and sadness in a tweet."
    )
    parser.add_argument(
        "tweet", nargs="*", help="Tweet text. Omit it to enter a tweet interactively."
    )
    args = parser.parse_args()
    text = " ".join(args.tweet).strip() if args.tweet else input("Tweet: ").strip()
    if not text:
        raise SystemExit("Please enter a non-empty tweet.")

    bundle = load_model_bundle()
    results = predict_emotions(text, bundle)
    emotion, confidence = results[0]

    print(f"\nPredicted emotion: {emotion} ({confidence:.1f}%)")
    print("Emotion percentages:")
    for label, percentage in results:
        print(f"  {label:<10} {percentage:5.1f}%")


if __name__ == "__main__":
    main()
