from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        groups[distribution].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return list[RagasResult]
    """
    if not answers:
        return []
    required = {"id", "distribution", "question", "answer", "contexts", "ground_truth"}
    for index, item in enumerate(answers):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"answers[{index}] is missing keys: {sorted(missing)}")
        if not isinstance(item["contexts"], list):
            raise ValueError(f"answers[{index}]['contexts'] must be a list")

    from src.m4_eval import evaluate_ragas
    raw = evaluate_ragas(
        [item["question"] for item in answers],
        [item["answer"] for item in answers],
        [item["contexts"] for item in answers],
        [item["ground_truth"] for item in answers],
    )
    per_question = raw.get("per_question", [])
    if len(per_question) != len(answers):
        raise RuntimeError(
            f"RAGAS returned {len(per_question)} rows for {len(answers)} answers"
        )
    return [
        RagasResult(
            question_id=item["id"], distribution=item["distribution"],
            question=item["question"], answer=item["answer"],
            contexts=item["contexts"], ground_truth=item["ground_truth"],
            faithfulness=float(score.faithfulness),
            answer_relevancy=float(score.answer_relevancy),
            context_precision=float(score.context_precision),
            context_recall=float(score.context_recall),
        )
        for item, score in zip(answers, per_question)
    ]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    output = []
    for rank, result in enumerate(sorted(results, key=lambda row: row.avg_score)[:10], 1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank, "question_id": result.question_id,
            "distribution": result.distribution, "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Mục tiêu: tìm ra distribution nào hay bị failure nhất và metric nào yếu nhất.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    distributions = ("factual", "multi_hop", "adversarial")
    matrix = {
        metric: {distribution: 0 for distribution in distributions}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        if result.distribution not in distributions:
            raise ValueError(f"Unknown distribution: {result.distribution!r}")
        matrix[result.worst_metric][result.distribution] += 1

    distribution_counts = {
        distribution: sum(matrix[metric][distribution] for metric in matrix)
        for distribution in distributions
    }
    distribution_averages = {
        distribution: (
            sum(result.avg_score for result in results if result.distribution == distribution)
            / distribution_counts[distribution]
            if distribution_counts[distribution] else 1.0
        )
        for distribution in distributions
    }
    metric_counts = {
        metric: sum(by_distribution.values())
        for metric, by_distribution in matrix.items()
    }
    # Every question contributes exactly one worst metric, so raw column totals
    # merely mirror distribution sizes. The lowest average quality is the useful
    # definition of the distribution with the most severe failures.
    dominant_distribution = min(distribution_averages, key=distribution_averages.get)
    dominant_metric = max(metric_counts, key=metric_counts.get)
    insight = (
        f"Distribution '{dominant_distribution}' có chất lượng thấp nhất "
        f"(avg={distribution_averages[dominant_distribution]:.3f}); metric "
        f"'{dominant_metric}' là điểm yếu chủ đạo ({metric_counts[dominant_metric]} câu). "
        f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}"
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_distribution,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
