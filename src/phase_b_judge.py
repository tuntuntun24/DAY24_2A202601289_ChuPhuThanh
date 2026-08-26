from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

@lru_cache(maxsize=256)
def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    prompt = f"""Câu hỏi:
{question}

Câu trả lời A:
{answer_a}

Câu trả lời B:
{answer_b}

Hãy đánh giá độc lập độ chính xác, đầy đủ và súc tích. Không ưu tiên câu trả lời
xuất hiện trước hoặc câu dài hơn. Trả về JSON có đúng cấu trúc:
{{"winner":"A|B|tie","reasoning":"giải thích ngắn gọn","scores":{{"A":0.0,"B":0.0}}}}
Điểm nằm trong [0, 1]. Chỉ chọn tie khi chất lượng thực sự tương đương."""

    if not OPENAI_API_KEY:
        return {
            "winner": "tie",
            "reasoning": "Không có OPENAI_API_KEY; dùng kết quả bảo thủ.",
            "scores": {"A": 0.5, "B": 0.5},
        }

    try:
        from openai import OpenAI

        response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia đánh giá câu trả lời RAG về chính sách nhân sự. "
                        "Chỉ trả lời JSON hợp lệ và không thiên vị vị trí."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        winner = str(parsed.get("winner", "tie")).strip()
        if winner.lower() == "tie":
            winner = "tie"
        else:
            winner = winner.upper()
        if winner not in {"A", "B", "tie"}:
            winner = "tie"

        raw_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
        scores = {}
        for label in ("A", "B"):
            try:
                value = float(raw_scores.get(label, 0.5))
            except (TypeError, ValueError):
                value = 0.5
            scores[label] = round(min(1.0, max(0.0, value)), 4)
        reasoning = str(parsed.get("reasoning") or "Hai câu trả lời có chất lượng tương đương.").strip()
        return {"winner": winner, "reasoning": reasoning, "scores": scores}
    except Exception as exc:
        return {
            "winner": "tie",
            "reasoning": f"Judge API không khả dụng: {type(exc).__name__}.",
            "scores": {"A": 0.5, "B": 0.5},
        }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    consistent = pass1["winner"] == winner_pass2
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=pass1["winner"] if consistent else "tie",
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have equal lengths")
    if not judge_labels:
        return 0.0
    if any(label not in {0, 1} for label in judge_labels + human_labels):
        raise ValueError("Cohen's kappa labels must be binary values 0 or 1")

    n = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / n
    expected = (
        judge_labels.count(1) / n * human_labels.count(1) / n
        + judge_labels.count(0) / n * human_labels.count(0) / n
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))


@lru_cache(maxsize=128)
def absolute_judge_label(question: str, answer: str, reference: str) -> dict:
    """Judge whether one answer is acceptable against a trusted reference."""
    if not OPENAI_API_KEY:
        return {"label": 0, "score": 0.0, "reasoning": "OPENAI_API_KEY is missing."}
    prompt = f"""Question: {question}

Candidate answer: {answer}

Trusted reference: {reference}

Classify the candidate as 1 (good) or 0 (bad). Judge only what the question
explicitly asks. A good answer must contain the correct requested facts, but it
may be much shorter than the reference and does not need supporting rationale,
policy-version wording, calculation steps, percentages, approval details, or
other conditions unless the question explicitly asks for them. For example, an
answer giving the correct repayment amount is good even if it does not say
"100%"; an answer giving the correct leave rule is good even if it omits an
unasked approval condition. A wrong requested fact, missing requested subpart,
or unsafe recommendation makes it bad. Return JSON only:
{{"label":0,"score":0.0,"reasoning":"short explanation"}}"""
    try:
        from openai import OpenAI

        response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict, position-independent RAG evaluator."},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        label = 1 if parsed.get("label") in {1, True, "1"} else 0
        try:
            score = min(1.0, max(0.0, float(parsed.get("score", label))))
        except (TypeError, ValueError):
            score = float(label)
        return {
            "label": label,
            "score": round(score, 4),
            "reasoning": str(parsed.get("reasoning") or "No reasoning returned."),
        }
    except Exception as exc:
        return {"label": 0, "score": 0.0, "reasoning": f"Judge failed: {type(exc).__name__}."}


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    position_bias_count = sum(not result.position_consistent for result in judge_results)
    decisive = [result for result in judge_results if result.final_winner in {"A", "B"}]
    a_wins_a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in decisive
    )
    b_wins_b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in decisive
    )
    position_rate = position_bias_count / total if total else 0.0
    verbosity_rate = (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    if not total:
        interpretation = "Chưa có dữ liệu để đánh giá bias."
    elif position_rate > 0.3:
        interpretation = "Position bias cao; cần giữ swap-and-average trong pipeline đánh giá."
    elif verbosity_rate > 0.6:
        interpretation = "Position bias thấp nhưng verbosity bias cao; cần siết rubric về tính súc tích."
    else:
        interpretation = "Judge ổn định; chưa thấy bias vượt ngưỡng cảnh báo."
    return {
        "total_judged": total,
        "position_bias_rate": round(position_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_rate, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        references = {item["id"]: item["ground_truth"] for item in json.load(f)}

    print(f"Running swap-and-average on {len(human_data)} human-labelled answers...")
    results = []
    absolute_results = []
    for index, item in enumerate(human_data, 1):
        result = swap_and_average(
            item["question"], item["model_answer"], references[item["question_id"]]
        )
        results.append(result)
        absolute = absolute_judge_label(
            item["question"], item["model_answer"], references[item["question_id"]]
        )
        absolute_results.append(absolute)
        print(f"  [{index}/{len(human_data)}] final={result.final_winner}, label={absolute['label']}")

    human_labels = [item["human_label"] for item in human_data]
    judge_labels = [item["label"] for item in absolute_results]
    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(results)
    agreement_rows = [
        {
            "question_id": item["question_id"],
            "question": item["question"],
            "human_label": human,
            "judge_label": judge,
            "agree": human == judge,
        }
        | {
            "judge_score": absolute["score"],
            "judge_reasoning": absolute["reasoning"],
        }
        for item, human, judge, absolute
        in zip(human_data, human_labels, judge_labels, absolute_results)
    ]
    report = {
        "model": JUDGE_MODEL,
        "cohen_kappa": round(kappa, 4),
        "human_agreement": agreement_rows,
        "bias": bias,
        "pairwise_results": [asdict(result) for result in results],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Judge report saved -> reports/judge_results.json")
