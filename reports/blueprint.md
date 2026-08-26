# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Chu Phú Thành  
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

~~~
User Input
    │
    ▼ (regex fallback: 0.03 ms P95 trong lần đo đã lưu)
[PII Scan — Presidio khi khả dụng, regex fallback khi thiếu dependency]
    │ block: VN_CCCD / VN_PHONE / EMAIL_ADDRESS
    │ action: reject + security log
    ▼ (fallback path: 0.02 ms P95 trong lần đo đã lưu)
[Deterministic fast rail → NeMo Input Rail khi khả dụng]
    │ block: off-topic / jailbreak / prompt injection / PII request
    │ action: refuse + reason; fail closed nếu NeMo unavailable
    ▼
[RAG Pipeline (Day 18)]
    │ Hierarchical chunks → BM25 + BGE-M3 → RRF → rerank → GPT-4o-mini
    ▼
[NeMo Output Rail + PII scan]
    │ flag: PII / sensitive content
    │ action: replace with safe response + log
    ▼
User Response
~~~

---

## Latency Budget

Task 12 đo riêng guard path; RAG generation không nằm trong phép đo này. Tại môi trường
đã sinh `guard_results.json`, package Presidio và NeMo chưa khả dụng, nên các số dưới đây
là latency của **regex/deterministic fallback path**, không phải latency gọi NeMo/LLM thật.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| PII scan (regex fallback) | 0.01 | 0.03 | 0.03 | <10 ms |
| Input rail (fallback, chưa gọi NeMo) | 0.01 | 0.02 | 0.02 | <300 ms |
| RAG Pipeline | Không đo riêng | Không đo riêng | Không đo riêng | <2000 ms |
| NeMo Output Rail | Không đo riêng | Không đo riêng | Không đo riêng | <300 ms |
| **Total Guard** | **0.02** | **0.05** | **0.05** | **<500 ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Fallback guard P95 đạt 0.05 ms, thấp hơn budget 500 ms. Kết quả này chỉ
xác nhận budget cho fast path; CI production phải cài đủ dependencies, cấu hình API key
và đo lại NeMo path trước khi dùng con số này làm release gate. Khi NeMo chưa khả dụng,
rail trả về trạng thái blocked thay vì cho request/response chưa kiểm tra đi qua.

---

## CI/CD Gates (phải pass trước khi merge to main)

~~~yaml
- name: Unit tests
  run: pytest tests/ -v

- name: RAGAS quality gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Judge agreement gate
  run: python src/phase_b_judge.py
  env:
    MIN_COHEN_KAPPA: 0.60

- name: Guardrail gate
  run: python src/phase_c_guard.py
  # yêu cầu adversarial pass rate >= 90% (18/20)

- name: Guard latency gate
  run: python src/phase_c_guard.py
  # yêu cầu total guard P95 < 500 ms
~~~

Kết quả hiện tại: unit tests 40/40; overall RAGAS 0.7212; Cohen's κ 0.800;
adversarial 20/20 trên active fallback path; fallback guard P95 0.05 ms. Faithfulness
tổng hợp chỉ đạt 0.615 nên quality gate 0.75 hiện phải fail và chặn merge. NeMo gate
cũng chưa được xem là đạt cho đến khi chạy lại với `nemoguardrails` và API key thật.

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---:|---|
| RAGAS faithfulness (daily sample) | <0.70 | Page on-call, phân tích claim không có citation |
| Multi-hop faithfulness | <0.60 | Review retrieval/prompt và bổ sung decomposition |
| Adversarial block rate | <90% | Bổ sung attack patterns và chạy regression suite |
| Guard P95 latency | >500 ms | Tách fast rail, kiểm tra model/API bottleneck |
| Position bias rate | >30% | Bắt buộc swap-and-average |
| PII detected count | spike >10/hour | Security alert và điều tra nguồn request |

Không log raw PII. Log loại entity, layer chặn, latency, policy version và trace ID đã băm.

---

## Kết quả thực tế từ Lab

| Chỉ số | Kết quả |
|---|---:|
| RAGAS avg_score (50q) | 0.7212 |
| RAGAS faithfulness tổng hợp | 0.6150 |
| Worst metric | faithfulness |
| Dominant failure distribution | multi_hop (avg 0.6070) |
| Cohen's κ | 0.800 |
| Position bias rate | 20% |
| Adversarial pass rate | 20/20 (100%, fallback path) |
| Guard P95 latency | 0.05 ms (fallback path; chưa phải NeMo/LLM) |

---

## Nhận xét & Cải tiến

Guard stack hoạt động tốt trên bộ regression hiện tại: regex và deterministic rail phát
hiện đủ CCCD, số điện thoại, email và chặn 20/20 input adversarial. Kết quả này chưa chứng
minh NeMo hoạt động vì lần chạy báo cáo dùng fallback. Điểm yếu lớn nhất nằm ở RAG
multi-hop, đặc biệt faithfulness 0.2875, cho thấy câu trả lời tổng hợp nhiều policy dễ tạo
claim không được context hỗ trợ. Trước khi production cần cài/kiểm thử NeMo thật, thêm
query decomposition, lọc phiên bản policy hiện hành, citation theo từng claim và đo riêng
output rail cùng end-to-end RAG P95 dưới tải đồng thời. LLM Judge đạt κ 0.800 nhưng vẫn
có 20% position inconsistency, vì vậy swap-and-average phải được giữ lại.
