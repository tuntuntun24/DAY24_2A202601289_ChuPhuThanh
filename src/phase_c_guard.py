from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

_presidio_setup_attempted = False
_presidio_pair = None
_nemo_setup_attempted = False
_nemo_instance = None


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

@lru_cache(maxsize=1)
def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NoOpNlpEngine
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern(
            "Email address",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            0.95,
        )],
    )

    registry = RecognizerRegistry(supported_languages=[PRESIDIO_LANGUAGE])
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(email_recognizer)

    nlp_engine = NoOpNlpEngine([
        {"lang_code": PRESIDIO_LANGUAGE, "model_name": "regex-only"}
    ])
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=[PRESIDIO_LANGUAGE],
    )
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def _get_presidio_safely():
    global _presidio_pair, _presidio_setup_attempted
    if not _presidio_setup_attempted:
        _presidio_setup_attempted = True
        try:
            _presidio_pair = setup_presidio()
        except Exception:
            _presidio_pair = None
    return _presidio_pair


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # These explicit Vietnamese recognizers also provide a dependency-safe
    # fallback when Presidio's NLP model is not available.
    patterns = (
        ("EMAIL_ADDRESS", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), 0.95),
        ("VN_PHONE", re.compile(r"\b0[3-9]\d{8}\b"), 0.90),
        ("VN_CCCD", re.compile(r"\b\d{12}\b"), 0.90),
        ("VN_CCCD", re.compile(r"\b\d{9}\b"), 0.70),
    )
    found = []
    presidio_used = False
    for entity_type, pattern, score in patterns:
        for match in pattern.finditer(text):
            found.append((entity_type, match.start(), match.end(), score))

    if analyzer is None:
        default_pair = _get_presidio_safely()
        if default_pair:
            analyzer, default_anonymizer = default_pair
            anonymizer = anonymizer or default_anonymizer
    if analyzer is not None:
        try:
            for result in analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE):
                found.append((result.entity_type, result.start, result.end, float(result.score)))
            presidio_used = True
        except Exception:
            pass

    # Deduplicate identical spans and prefer the higher-confidence entity.
    unique = {}
    for entity_type, start, end, score in found:
        key = (start, end)
        current = unique.get(key)
        if current is None or score > current[3]:
            unique[key] = (entity_type, start, end, score)
    matches = sorted(unique.values(), key=lambda item: (item[1], item[2]))
    entities = [
        {
            "type": entity_type,
            "text": text[start:end],
            "score": round(score, 3),
            "start": start,
            "end": end,
        }
        for entity_type, start, end, score in matches
    ]
    anonymized_text = text
    for entity_type, start, end, _score in reversed(matches):
        anonymized_text = anonymized_text[:start] + f"<{entity_type}>" + anonymized_text[end:]
    return {
        "has_pii": bool(entities),
        "entities": entities,
        "anonymized": anonymized_text,
        "engine": "presidio" if presidio_used else "regex_fallback",
    }


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

@lru_cache(maxsize=1)
def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


def _get_nemo_safely():
    global _nemo_instance, _nemo_setup_attempted
    if not _nemo_setup_attempted:
        _nemo_setup_attempted = True
        try:
            _nemo_instance = setup_nemo_rails()
        except Exception:
            _nemo_instance = None
    return _nemo_instance


def _response_text(response) -> str:
    """Normalize NeMo responses across supported library versions."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("content") or response.get("response") or response)
    return str(response)


def _local_input_violation(text: str) -> str | None:
    """Fast deterministic rail for explicit attacks and known off-topic intents."""
    normalized = " ".join(text.casefold().split())
    categories = {
        "prompt_injection": (
            "bỏ qua tất cả", "bỏ qua policy", "ignore your system", "ignore previous",
            "forget all", "system override", "admin command", "priority 1",
            "system instructions", "system prompt", "training data", "đóng vai",
            "pretend you are dan", "không có giới hạn", "không giới hạn",
        ),
        "pii_request": (
            "cho tôi biết cccd", "số điện thoại của nhân viên", "thông tin nhân viên",
            "thông tin cá nhân", "tiết lộ bảng lương", "tiết lộ lương",
            "employee salaries", "employee records", "confidential hr data",
            "confidential employee data", "mật khẩu admin", "mật khẩu hệ thống",
        ),
        "off_topic": (
            "bài thơ", "nấu phở", "nấu ăn", "bitcoin", "ethereum", "phương trình",
            "recommend", "bộ phim", "thời tiết", "giá cổ phiếu", "tấn công mạng",
        ),
    }
    for category, phrases in categories.items():
        if any(phrase in normalized for phrase in phrases):
            return category
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    reason = _local_input_violation(text)
    if reason:
        return {
            "allowed": False,
            "blocked_reason": reason,
            "response": "Xin lỗi, yêu cầu đã bị chặn bởi chính sách an toàn.",
            "engine": "deterministic_input_rail",
        }

    try:
        rails = rails or _get_nemo_safely()
        if rails is None:
            raise RuntimeError("NeMo could not be initialized")
        raw_response = await rails.generate_async(messages=[{"role": "user", "content": text}])
        response = _response_text(raw_response)
        blocked = any(keyword in response.casefold() for keyword in (
            "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
        ))
        return {
            "allowed": not blocked,
            "blocked_reason": "nemo_input_rail" if blocked else None,
            "response": response,
            "engine": "nemo_input_rail",
        }
    except Exception as exc:
        # Unknown inputs fail closed when the semantic rail is unavailable.
        return {
            "allowed": False,
            "blocked_reason": "nemo_unavailable",
            "response": (
                f"NeMo unavailable ({type(exc).__name__}); "
                "request blocked safely until the rail recovers."
            ),
            "engine": "fallback_no_nemo",
        }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    pii_result = pii_scan(answer)
    sensitive = any(phrase in answer.casefold() for phrase in (
        "mật khẩu hệ thống", "mật khẩu admin", "toàn bộ bảng lương",
        "dữ liệu bí mật", "confidential employee data",
    ))
    if pii_result["has_pii"] or sensitive:
        return {
            "safe": False,
            "flagged_reason": "pii_output" if pii_result["has_pii"] else "sensitive_output",
            "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm này. Vui lòng liên hệ phòng Nhân sự.",
            "engine": "pii_scan" if pii_result["has_pii"] else "deterministic_output_rail",
        }

    try:
        rails = rails or _get_nemo_safely()
        if rails is None:
            raise RuntimeError("NeMo could not be initialized")
        raw_response = await rails.generate_async(messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        response = _response_text(raw_response)
        flagged = any(keyword in response.casefold() for keyword in (
            "không thể cung cấp", "i cannot provide", "thông tin nhạy cảm",
        ))
        return {
            "safe": not flagged,
            "flagged_reason": "nemo_output_rail" if flagged else None,
            "final_answer": response if flagged else answer,
            "engine": "nemo_output_rail",
        }
    except Exception:
        return {
            "safe": False,
            "flagged_reason": "nemo_unavailable",
            "final_answer": (
                "Hệ thống kiểm tra an toàn tạm thời không khả dụng. "
                "Vui lòng thử lại sau."
            ),
            "engine": "fallback_no_nemo",
        }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"
                guard_engine = pii_result["engine"]
            else:
                rail_result = await check_input_rail(item["input"], rails)
                guard_engine = rail_result["engine"]
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"], "category": item["category"],
                "input": item["input"], "expected": item["expected"],
                "actual": actual, "blocked_by": blocked_by,
                "guard_engine": guard_engine,
                "passed": actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(result["passed"] for result in results)
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    if n_runs < 0:
        raise ValueError("n_runs must be non-negative")
    if not test_inputs or n_runs == 0:
        zero = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "presidio_ms": dict(zero), "nemo_ms": dict(zero), "total_ms": dict(zero),
            "latency_budget_ok": True, "budget_ms": LATENCY_BUDGET_P95_MS,
            "presidio_mode": "not_measured",
            "input_rail_mode": "not_measured",
            "measurement_note": "No latency samples were collected.",
        }

    presidio_times, nemo_times, total_times = [], [], []
    presidio_modes, input_rail_modes = [], []

    async def _measure():
        for index in range(n_runs):
            text = test_inputs[index % len(test_inputs)]
            start = time.perf_counter()
            pii_result = pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            rail_result = await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - start) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)
            presidio_modes.append(pii_result["engine"])
            input_rail_modes.append(rail_result["engine"])

    asyncio.run(_measure())

    def percentiles(values):
        ordered = sorted(values)
        def nearest_rank(percentile):
            index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
            return round(ordered[index], 2)
        return {"p50": nearest_rank(0.50), "p95": nearest_rank(0.95), "p99": nearest_rank(0.99)}

    total_percentiles = percentiles(total_times)
    presidio_mode = ",".join(sorted(set(presidio_modes)))
    input_rail_mode = ",".join(sorted(set(input_rail_modes)))
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms": percentiles(nemo_times),
        "total_ms": total_percentiles,
        "latency_budget_ok": total_percentiles["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
        "presidio_mode": presidio_mode,
        "input_rail_mode": input_rail_mode,
        "measurement_note": (
            "nemo_ms is actual NeMo latency only when input_rail_mode is "
            "'nemo_input_rail'; otherwise it measures the deterministic/fallback path."
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    passed = sum(1 for r in results if r["passed"])

    # Task 12: P95 latency
    sample_inputs = [
        "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
        "Chính sách bảo hiểm sức khỏe dành cho nhân viên là gì?",
        "Quy trình xin tạm ứng được quy định như thế nào?",
        "Khi làm việc từ xa, nhân viên phải dùng VPN nào?",
        "Phụ cấp ăn trưa hàng tháng là bao nhiêu?",
    ]
    # Safe initialization keeps the script runnable in a minimal environment;
    # the report explicitly records whether real engines or fallbacks ran.
    presidio_pair = _get_presidio_safely()
    analyzer, anonymizer = presidio_pair if presidio_pair else (None, None)
    rails = _get_nemo_safely()
    latency = measure_p95_latency(
        sample_inputs,
        n_runs=10,
        rails=rails,
        analyzer=analyzer,
        anonymizer=anonymizer,
    )
    print(f"\nLatency P95 — PII scan: {latency['presidio_ms']['p95']}ms | "
          f"Input rail ({latency['input_rail_mode']}): {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    output_demo = asyncio.run(check_output_rail(
        "Hãy cung cấp CCCD của nhân viên A.",
        "CCCD của nhân viên là 034095001234.",
    ))
    report = {
        "runtime": {
            "presidio_mode": latency["presidio_mode"],
            "input_rail_mode": latency["input_rail_mode"],
            "nemo_available": rails is not None,
            "note": (
                "Adversarial pass rate covers the active guard path. "
                "It validates NeMo itself only when nemo_available is true."
            ),
        },
        "adversarial_summary": {
            "total": len(results),
            "passed": passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
        },
        "adversarial_results": results,
        "latency": latency,
        "output_rail_sample": output_demo,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Guard report saved -> reports/guard_results.json")
