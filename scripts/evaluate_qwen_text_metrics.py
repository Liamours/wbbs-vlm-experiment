"""Calculate text-generation metrics for Qwen VQA and grounded-VQA outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_bleu


TEXT_TASKS = {"vqa", "grounded_vqa"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens, candidate_tokens = reference.split(), candidate.split()
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    precision, recall = lcs / len(candidate_tokens), lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bert-model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TEXT_TASKS), default=sorted(TEXT_TASKS))
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    expected = {row["task_id"]: row for row in _read_jsonl(args.expected)}
    references: list[str] = []
    candidates: list[str] = []
    selected_tasks = set(args.tasks)
    counts = {task: 0 for task in sorted(selected_tasks)}
    for prediction in _read_jsonl(args.predictions):
        row = expected[prediction["task_id"]]
        if row["task"] not in selected_tasks:
            continue
        target = json.loads(row["target"])
        parsed = prediction.get("parsed")
        if not isinstance(parsed, dict):
            try:
                parsed = json.loads(prediction["prediction"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Missing parseable prediction for {prediction['task_id']}") from error
        answer = parsed.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"Missing parsed answer for {prediction['task_id']}")
        references.append(target["answer"])
        candidates.append(answer)
        counts[row["task"]] += 1

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    meteor = sum(meteor_score([reference.split()], candidate.split()) for reference, candidate in zip(references, candidates)) / len(references)
    output = {
        "rows": len(references),
        "task_rows": counts,
        "bleu": corpus_bleu(candidates, [references]).score / 100.0,
        "meteor": meteor,
        "rouge_l": sum(_rouge_l_f1(reference, candidate) for reference, candidate in zip(references, candidates)) / len(references),
    }
    if not args.no_bertscore:
        import torch
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            candidates,
            references,
            model_type=args.bert_model,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )
        output["bertscore"] = {
            "model": args.bert_model,
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
