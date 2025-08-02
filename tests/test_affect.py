from pathlib import Path

from mmstory.affect import EmotionTagger, TextSentimentClassifier

DATA_DIR = Path(__file__).parent / "data"


def _read(name: str) -> str:
    return (DATA_DIR / name).read_text()


def test_sentiment_classifier_positive_prose() -> None:
    prose = _read("prose.txt")
    result = TextSentimentClassifier().analyze(prose)
    assert result.label == "positive"
    assert result.scores["positive"] > result.scores["negative"]
    assert result.confidence >= 0.9


def test_sentiment_classifier_negative_brief() -> None:
    brief = _read("brief.txt")
    result = TextSentimentClassifier().analyze(brief)
    assert result.label == "negative"
    assert result.scores["negative"] > result.scores["positive"]
    assert "paranoid" in result.evidence


def test_emotion_tagger_highlights_fear_in_prose() -> None:
    prose = _read("prose.txt")
    result = EmotionTagger().tag(prose)
    assert result.dominant == "fear"
    assert result.scores["fear"] >= result.scores["joy"]
    assert "paranoia" in result.evidence.get("fear", [])


def test_emotion_tagger_links_envy_to_anger() -> None:
    raw = _read("scene_plan.json")
    result = EmotionTagger().tag(raw)
    assert "envious" in result.evidence.get("anger", [])
    assert result.scores["anger"] > 0.0
