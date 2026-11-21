from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys
import textwrap
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Project.cli.preprocessing import (
    average_word2vec,
    identity_tokenizer,  # Imported so joblib can resolve the saved vectorizer tokenizer.
    join_token_lists,  # Imported so joblib can resolve the character feature transformer.
    tokenize_normalize,
)

MODEL_PATH = Path(__file__).resolve().parent / "emotion_model.joblib"


class TerminalUI:
    """Small ANSI interface with a plain-text fallback for redirected output."""

    PALETTE = {
        "accent": (115, 168, 255),
        "text": (236, 238, 241),
        "muted": (150, 164, 184),
        "rule": (64, 78, 102),
        "anger": (239, 112, 102),
        "joy": (123, 190, 142),
        "optimism": (241, 188, 91),
        "sadness": (120, 165, 235),
        "error": (255, 105, 105),
    }

    def __init__(self) -> None:
        self.use_color = (
            "NO_COLOR" not in os.environ
            and sys.stdout.isatty()
        )
        self.width = max(40, min(shutil.get_terminal_size((80, 24)).columns - 4, 72))
        if self.use_color:
            self._enable_windows_ansi()

    @staticmethod
    def _enable_windows_ansi() -> None:
        if os.name != "nt":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            stdout_handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(stdout_handle, mode.value | 0x0004)
        except (AttributeError, OSError):
            pass

    def paint(self, value: str, tone: str = "text", bold: bool = False) -> str:
        if not self.use_color:
            return value
        red, green, blue = self.PALETTE[tone]
        weight = "1;" if bold else ""
        return f"\033[{weight}38;2;{red};{green};{blue}m{value}\033[0m"

    def _box_row(self, value: str, tone: str = "text", bold: bool = False) -> None:
        content_width = self.width - 3
        shown = value[:content_width]
        padding = " " * max(0, content_width - len(shown))
        print(f"  \u2502 {self.paint(shown, tone, bold)}{padding}\u2502")

    def banner(self) -> None:
        rule = self.paint("\u2500" * (self.width - 2), "rule")
        print(f"  \u256d{rule}\u256e")
        self._box_row("TWEET EMOTION", "accent", bold=True)
        self._box_row("ANGER  \u00b7  JOY  \u00b7  OPTIMISM  \u00b7  SADNESS", "muted")
        print(f"  \u2570{rule}\u256f")
        print()

    def show_tweet(self, tweet: str) -> None:
        print(self.paint("  YOU", "muted", bold=True))
        display_text = " ".join(tweet.split())
        wrap_width = max(28, self.width - 8)
        marker = self.paint("\u203a", "accent", bold=True)
        for line in textwrap.wrap(display_text, width=wrap_width) or [""]:
            print(f"  {marker} {line}")
        print()

    def show_results(self, results: list[tuple[str, float]]) -> None:
        emotion, score = results[0]
        tone = emotion if emotion in self.PALETTE else "accent"
        print(self.paint("  PREDICTION", "muted", bold=True))
        marker = self.paint("\u25c6", tone, bold=True)
        emotion_name = self.paint(emotion.upper(), tone, bold=True)
        estimate = self.paint(f"{score:.1f}% strongest estimate", "text")
        print(f"  {marker} {emotion_name}  {estimate}")
        print()
        print(self.paint("  CLASS DISTRIBUTION", "muted", bold=True))

        columns = shutil.get_terminal_size((80, 24)).columns
        bar_width = max(10, min(24, columns - 48))
        for label, percentage in results:
            filled = max(0, min(bar_width, round(percentage / 100 * bar_width)))
            empty = bar_width - filled
            tone = label if label in self.PALETTE else "accent"
            bar = self.paint("\u2501" * filled, tone, bold=True)
            bar += self.paint("\u2500" * empty, "rule")
            print(f"  {label.title():<10} {bar}  {percentage:5.1f}%")
        print()
        print(self.paint("  Percentages are model estimates, not certainty.", "muted"))
        print()

    def error(self, message: str) -> None:
        marker = self.paint("!", "error", bold=True)
        print(f"\n  {marker} {message}", file=sys.stderr)


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(
            "Trained model not found. Run TED-NLP.ipynb through its final refit/export cell first."
        )
    try:
        return joblib.load(MODEL_PATH)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate anger, joy, optimism, and sadness in a tweet."
    )
    parser.add_argument(
        "tweet", nargs="*", help="Tweet text. Omit it for interactive mode."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ui = TerminalUI()
    ui.banner()

    try:
        bundle = load_model_bundle()
    except SystemExit as exc:
        ui.error(str(exc))
        return 1

    if args.tweet:
        tweet = " ".join(args.tweet).strip()
        if not tweet:
            ui.error("Please enter a non-empty tweet.")
            return 2
        try:
            ui.show_tweet(tweet)
            ui.show_results(predict_emotions(tweet, bundle))
        except SystemExit as exc:
            ui.error(str(exc))
            return 1
        return 0

    print("  Type a tweet to analyze. Use /help for commands or /exit to quit.\n")
    while True:
        try:
            prompt = ui.paint("  ? ", "accent", bold=True)
            tweet = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  " + ui.paint("See you next time.", "muted"))
            return 0

        if tweet.lower() in {"/exit", "/quit"}:
            print("\n  " + ui.paint("See you next time.", "muted"))
            return 0
        if tweet.lower() == "/help":
            print("\n  Enter a tweet to see its emotion estimates.")
            print("  /help  show this message")
            print("  /exit  close the session\n")
            continue
        if not tweet:
            continue

        try:
            ui.show_tweet(tweet)
            ui.show_results(predict_emotions(tweet, bundle))
        except SystemExit as exc:
            ui.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
