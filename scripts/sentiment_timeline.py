#!/usr/bin/env python3
"""
Plot sentiment timeline for generated prose + dialogue.

Usage:
    python scripts/sentiment_timeline.py prose.txt dialogue.txt output.png
"""
import sys
import matplotlib.pyplot as plt
from mmstory.affect import TextSentimentClassifier

def load_sentences(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    sentences = [segment.strip() for segment in text.replace("\n", " ").split(".") if segment.strip()]
    return sentences

def scores_for(sentences: list[str], classifier: TextSentimentClassifier) -> list[float]:
    values = []
    for sentence in sentences:
        result = classifier.analyze(sentence)
        score = result.scores["positive"] - result.scores["negative"]
        values.append(score)
    return values

def main(prose_path: str, dialogue_path: str, output_path: str) -> None:
    clf = TextSentimentClassifier()
    prose_sentences = load_sentences(prose_path)
    dialogue_sentences = load_sentences(dialogue_path)
    prose_scores = scores_for(prose_sentences, clf)
    dialogue_scores = scores_for(dialogue_sentences, clf)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prose_scores, marker="o", label="Prose Sentiment")
    ax.plot(range(len(prose_scores), len(prose_scores) + len(dialogue_scores)),
            dialogue_scores, marker="s", label="Dialogue Sentiment")
    ax.axhline(0.0, color="#888", linestyle="--", linewidth=1)
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Positive – Negative Score")
    ax.set_title("Emotional Trajectory Across Modalities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

if __name__ == "__main__":
    prose, dialogue, output = sys.argv[1], sys.argv[2], sys.argv[3]
    main(prose, dialogue, output)
