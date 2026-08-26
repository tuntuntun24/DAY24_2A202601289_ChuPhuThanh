from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")
    for index, item_contexts in enumerate(contexts):
        if not isinstance(item_contexts, list) or not all(
            isinstance(context, str) for context in item_contexts
        ):
            raise ValueError(f"contexts[{index}] must be a list of strings")

    fallback = _zero_results(questions, answers, contexts, ground_truths)
    if not questions:
        return fallback

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")

        # RAGAS 0.1.x uses these singleton metric objects and the singular
        # ``ground_truth`` dataset column.
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        records = result.to_pandas().to_dict(orient="records")

        per_question: list[EvalResult] = []
        for index, question in enumerate(questions):
            row = records[index] if index < len(records) else {}
            per_question.append(
                EvalResult(
                    question=question,
                    answer=answers[index],
                    contexts=contexts[index],
                    ground_truth=ground_truths[index],
                    faithfulness=_safe_score(row.get("faithfulness")),
                    answer_relevancy=_safe_score(row.get("answer_relevancy")),
                    context_precision=_safe_score(row.get("context_precision")),
                    context_recall=_safe_score(row.get("context_recall")),
                )
            )

        metric_names = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        aggregate = {
            metric: (
                sum(getattr(item, metric) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for metric in metric_names
        }
        return {**aggregate, "per_question": per_question}
    except Exception as exc:
        print(f"  [warning] RAGAS evaluation failed: {exc}")
        return fallback


def _safe_score(value) -> float:
    """Convert metric output to a finite float suitable for reports."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _zero_results(questions: list[str], answers: list[str],
                  contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Create a shape-preserving fallback when RAGAS cannot run."""
    per_question = [
        EvalResult(question, answer, item_contexts, ground_truth, 0.0, 0.0, 0.0, 0.0)
        for question, answer, item_contexts, ground_truth
        in zip(questions, answers, contexts, ground_truths)
    ]
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0 or not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM answer contains claims that are not supported by the retrieved context.",
            "Tighten the grounded-answer prompt, lower temperature, and require citations.",
            "Output incorrect → Context present → Claim unsupported → Generation failure",
        ),
        "answer_relevancy": (
            "The answer does not directly address the user's question.",
            "Improve the answer prompt and preserve the original query intent.",
            "Output off-topic → Context checked → Query intent lost → Prompt failure",
        ),
        "context_precision": (
            "Retrieved context contains too many irrelevant or low-ranked chunks.",
            "Improve reranking and apply metadata/version filters before generation.",
            "Output weak → Context noisy → Relevant chunks ranked low → Ranking failure",
        ),
        "context_recall": (
            "The retriever missed information required to answer the question.",
            "Improve chunking, hybrid retrieval, query expansion, or increase retrieval depth.",
            "Output incomplete → Context missing → Candidate not retrieved → Retrieval failure",
        ),
    }

    analyzed: list[dict] = []
    for result in eval_results:
        scores = {
            metric: _safe_score(getattr(result, metric))
            for metric in diagnostic_tree
        }
        worst_metric = min(scores, key=scores.get)
        average_score = sum(scores.values()) / len(scores)
        diagnosis, suggested_fix, error_tree = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "expected": result.ground_truth,
            "got": result.answer,
            "contexts": list(result.contexts),
            "worst_metric": worst_metric,
            "worst_metric_score": scores[worst_metric],
            "score": average_score,
            "diagnosis": diagnosis,
            "error_tree": error_tree,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda item: item["score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

