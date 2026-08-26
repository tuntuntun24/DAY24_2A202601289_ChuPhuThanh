"""Kiểm tra trước khi nộp — Lab 24: Eval + Guardrail Stack."""
from __future__ import annotations

import json
import os
import subprocess
import sys

STUDENT_NAME = "Chu Phú Thành"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✓" if condition else "✗"
    suffix = f" → {detail}" if detail and not condition else ""
    print(f"  [{icon}] {label}{suffix}")
    return condition


def main():
    print("=" * 60)
    print("CHECK LAB 24: Eval + Guardrail Stack")
    print("=" * 60)

    passed = total = 0

    # 1. Day 18 source files
    print("\n[1] Day 18 source files (copy từ lab của bạn):")
    day18_files = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    for f in day18_files:
        total += 1
        passed += check(f, os.path.exists(f), "missing — cp <Day18>/src/ .")

    # 2. Setup output
    print("\n[2] Setup output:")
    total += 1
    answers_ok = os.path.exists("answers_50q.json")
    passed += check("answers_50q.json", answers_ok, "run: python setup_answers.py")
    if answers_ok:
        with open("answers_50q.json", encoding="utf-8") as f:
            answers = json.load(f)
        total += 1
        passed += check("answers_50q.json has 50 entries", len(answers) == 50,
                        f"có {len(answers)} entries, cần 50")

    # 3. Phase implementations
    print("\n[3] Phase module implementations:")
    for module in ["src/phase_a_ragas.py", "src/phase_b_judge.py", "src/phase_c_guard.py"]:
        if os.path.exists(module):
            with open(module, encoding="utf-8") as f:
                content = f.read()
            todos = content.count("# TODO")
            total += 1
            passed += check(module, todos == 0, f"{todos} TODO(s) còn lại")

    # 4. Generated reports
    print("\n[4] Generated reports:")
    report_files = {
        "reports/ragas_50q.json":     "Phase A — python src/phase_a_ragas.py",
        "reports/judge_results.json": "Phase B — python src/phase_b_judge.py",
        "reports/guard_results.json": "Phase C — python src/phase_c_guard.py",
        "reports/blueprint.md":       "Task 13 — điền blueprint.md",
        "analysis/failure_clusters.md": "Phase A — phân tích failure clusters",
        "analysis/bias_report.md":      "Phase B — phân tích judge bias",
    }
    for path, hint in report_files.items():
        total += 1
        passed += check(path, os.path.exists(path), hint)

    markdown_reports = [
        "reports/blueprint.md",
        "analysis/failure_clusters.md",
        "analysis/bias_report.md",
    ]
    for path in markdown_reports:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            placeholders = ("[Họ Tên]", "[Ngày làm lab]", "___", "| ? |")
            total += 1
            passed += check(
                f"{path} filled in",
                len(content) > 500 and not any(marker in content for marker in placeholders),
                "Báo cáo còn trống hoặc còn placeholder",
            )
            total += 1
            passed += check(
                f"{path} has correct student name",
                f"**Sinh viên:** {STUDENT_NAME}" in content,
                f"Cần ghi đúng tên: {STUDENT_NAME}",
            )

    # 5. RAGAS report structure
    if os.path.exists("reports/ragas_50q.json"):
        print("\n[5] RAGAS report structure:")
        with open("reports/ragas_50q.json", encoding="utf-8") as f:
            report = json.load(f)
        for key in ("total_questions", "per_distribution", "failure_clusters", "bottom_10"):
            total += 1
            passed += check(f"ragas_50q.json['{key}']", key in report)
        if "total_questions" in report:
            total += 1
            passed += check("total_questions == 50", report["total_questions"] == 50,
                            f"có {report.get('total_questions')}, cần 50")
        expected_counts = {"factual": 20, "multi_hop": 20, "adversarial": 10}
        actual_counts = {
            name: values.get("count")
            for name, values in report.get("per_distribution", {}).items()
        }
        total += 1
        passed += check(
            "RAGAS distribution counts == 20/20/10",
            actual_counts == expected_counts,
            f"nhận được {actual_counts}",
        )
        bottom = report.get("bottom_10", [])
        total += 1
        passed += check(
            "RAGAS bottom_10 has 10 sorted rows",
            len(bottom) == 10
            and [row.get("avg_score") for row in bottom]
            == sorted(row.get("avg_score") for row in bottom),
        )
        matrix = report.get("failure_clusters", {}).get("matrix", {})
        matrix_total = sum(sum(by_dist.values()) for by_dist in matrix.values()) if matrix else 0
        total += 1
        passed += check("failure matrix accounts for all 50 questions", matrix_total == 50,
                        f"matrix total = {matrix_total}")

    # 6. Judge report consistency
    if os.path.exists("reports/judge_results.json"):
        print("\n[6] Judge report consistency:")
        with open("reports/judge_results.json", encoding="utf-8") as f:
            judge_report = json.load(f)
        pairwise = judge_report.get("pairwise_results", [])
        agreement = judge_report.get("human_agreement", [])
        bias = judge_report.get("bias", {})
        kappa = judge_report.get("cohen_kappa")
        total += 1
        passed += check("judge report has 10 pairwise results", len(pairwise) == 10)
        total += 1
        passed += check("judge report has 10 human comparisons", len(agreement) == 10)
        total += 1
        passed += check("Cohen's kappa is in [-1, 1]",
                        isinstance(kappa, (int, float)) and -1 <= kappa <= 1)
        inconsistent = sum(not row.get("position_consistent", False) for row in pairwise)
        total += 1
        passed += check(
            "position bias count matches pairwise results",
            bias.get("position_bias_count") == inconsistent,
            f"report={bias.get('position_bias_count')}, calculated={inconsistent}",
        )

    # 7. Guard report consistency
    if os.path.exists("reports/guard_results.json"):
        print("\n[7] Guard report consistency:")
        with open("reports/guard_results.json", encoding="utf-8") as f:
            guard_report = json.load(f)
        guard_results = guard_report.get("adversarial_results", [])
        guard_summary = guard_report.get("adversarial_summary", {})
        actual_passed = sum(bool(row.get("passed")) for row in guard_results)
        total += 1
        passed += check("guard report has 20 adversarial results", len(guard_results) == 20)
        total += 1
        passed += check(
            "guard summary matches detailed results",
            guard_summary.get("total") == len(guard_results)
            and guard_summary.get("passed") == actual_passed,
        )
        total += 1
        passed += check("adversarial pass rate >= 75%",
                        len(guard_results) > 0 and actual_passed / len(guard_results) >= 0.75)
        runtime = guard_report.get("runtime", {})
        latency = guard_report.get("latency", {})
        total += 1
        passed += check(
            "guard report records actual runtime modes",
            bool(runtime.get("presidio_mode")) and bool(runtime.get("input_rail_mode"))
            and runtime.get("presidio_mode") == latency.get("presidio_mode")
            and runtime.get("input_rail_mode") == latency.get("input_rail_mode"),
            "Chạy lại python src/phase_c_guard.py để ghi runtime metadata",
        )

    # 8. Test suite
    print("\n[8] Test suite:")
    result = subprocess.run(
        ["pytest", "tests/", "--tb=short", "-q"],
        capture_output=True, text=True,
    )
    tests_ok = result.returncode == 0
    total += 1
    passed += check("pytest tests/ passes", tests_ok, "run: pytest tests/ -v để xem chi tiết")
    if not tests_ok:
        lines = (result.stdout + result.stderr).strip().split("\n")
        print("\n" + "\n".join(lines[-20:]))

    print(f"\n{'='*60}")
    print(f"Score: {passed}/{total} checks passed")
    if passed == total:
        print("✓ Sẵn sàng nộp bài!")
    else:
        print(f"✗ Còn {total - passed} vấn đề cần giải quyết.")


if __name__ == "__main__":
    main()
