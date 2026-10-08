from __future__ import annotations

from typing import Any


def evaluate_text_answers(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Free-text generation quality for the 'answer' field. pairs = (predicted, reference)."""

    if not pairs:
        return {"rows": 0, "bleu": None, "meteor": None, "rouge_l": None, "bertscore_f1": None}
    hypotheses = [hypothesis for hypothesis, _ in pairs]
    references = [reference for _, reference in pairs]
    return {
        "rows": len(pairs),
        "bleu": _bleu(hypotheses, references),
        "meteor": _meteor(hypotheses, references),
        "rouge_l": _rouge_l(hypotheses, references),
        "bertscore_f1": _bertscore_f1(hypotheses, references),
    }


def _bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu

    return round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 4)


def _meteor(hypotheses: list[str], references: list[str]) -> float:
    from nltk.translate.meteor_score import meteor_score

    _ensure_nltk_data()
    scores = [meteor_score([reference.split()], hypothesis.split()) for hypothesis, reference in zip(hypotheses, references)]
    return round(sum(scores) / len(scores), 6)


def _rouge_l(hypotheses: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(reference, hypothesis)["rougeL"].fmeasure for hypothesis, reference in zip(hypotheses, references)]
    return round(sum(scores) / len(scores), 6)


def _bertscore_f1(hypotheses: list[str], references: list[str]) -> float:
    from bert_score import score

    # bert-base-uncased is already cached locally; rescale_with_baseline needs an
    # internet-fetched calibration file, so it stays off and scores are raw F1.
    _, _, f1 = score(hypotheses, references, model_type="bert-base-uncased", lang="en", rescale_with_baseline=False, verbose=False)
    return round(f1.mean().item(), 6)


_nltk_data_ready = False


def _ensure_nltk_data() -> None:
    global _nltk_data_ready
    if _nltk_data_ready:
        return
    import nltk

    for resource, path in (("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_data_ready = True
