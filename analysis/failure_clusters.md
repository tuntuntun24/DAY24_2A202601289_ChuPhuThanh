# Failure Cluster Analysis — Phase A

**Sinh viên:** Chu Phú Thành  
**Ngày:** 26/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

Nguồn số liệu: `reports/ragas_50q.json` (50 câu; factual 20, multi-hop 20,
adversarial 10). Các giá trị dưới đây đã được đối chiếu lại bằng trung bình có trọng số.

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.9000 | 0.2875 | 0.7000 |
| answer_relevancy | 0.8383 | 0.5447 | 0.6739 |
| context_precision | 0.8333 | 0.8750 | 0.8000 |
| context_recall | 0.8333 | 0.7208 | 0.5833 |
| **avg_score** | **0.8513** | **0.6070** | **0.6893** |

Overall avg_score của 50 câu là **0.7212**. Factual mạnh nhất; multi-hop yếu nhất.

---

## 2. Bottom 10 Questions

| Rank | ID | Distribution | Question (tóm tắt) | avg_score | worst_metric |
|---:|---:|---|---|---:|---|
| 1 | 40 | multi_hop | Thử việc tháng 3 phát hiện vi phạm bảo mật | 0.0000 | faithfulness |
| 2 | 49 | adversarial | So sánh phép năm v2023 với policy hiện hành | 0.1667 | faithfulness |
| 3 | 22 | multi_hop | Laptop 30 triệu: phê duyệt và yêu cầu CNTT | 0.2083 | faithfulness |
| 4 | 5 | factual | Thiết bị 55 triệu cần ai phê duyệt | 0.2083 | faithfulness |
| 5 | 6 | factual | Cấp độ phân loại của thông tin lương | 0.2500 | faithfulness |
| 6 | 50 | adversarial | Manager có được dùng VPN cá nhân khi WFH | 0.2917 | faithfulness |
| 7 | 37 | multi_hop | Tự xóa malware và chia sẻ sự cố trên Slack | 0.3333 | faithfulness |
| 8 | 25 | multi_hop | Lương thử việc Junior cao nhất | 0.3750 | faithfulness |
| 9 | 21 | multi_hop | Senior 9 năm: phép năm và khoảng lương | 0.3750 | faithfulness |
| 10 | 33 | multi_hop | Manager 12 năm: phụ cấp và phép v2024 | 0.3750 | faithfulness |

Bottom 10 có 6 multi-hop, 2 adversarial và 2 factual.

---

## 3. Failure Cluster Matrix

Mỗi câu được gán vào metric có điểm thấp nhất.

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 2 | 16 | 3 | **21** |
| answer_relevancy | 10 | 2 | 1 | **13** |
| context_precision | 7 | 0 | 1 | **8** |
| context_recall | 1 | 2 | 5 | **8** |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** multi_hop  
**Dominant metric:** faithfulness

Multi-hop có avg_score thấp nhất (0.6070) và 16/20 câu nhận faithfulness là metric yếu nhất. Retrieval vẫn có context precision cao (0.8750), nên lỗi chủ đạo không phải lấy quá nhiều tài liệu rác mà là tổng hợp/tính toán chưa được grounded theo từng claim. Các câu thường yêu cầu kết hợp ngưỡng tiền, thâm niên, version policy hoặc phép tính pro-rata; prompt generation hiện chưa bắt buộc trình bày bằng chứng và phép tính. Adversarial yếu chủ yếu ở context recall 0.5833, đặc biệt khi phải phân biệt policy cũ với bản hiện hành.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Claim tổng hợp/tính toán không được context hỗ trợ rõ | Query decomposition; citation theo claim; temperature 0; kiểm tra số học |
| context_recall | Thiếu tài liệu hoặc sai version policy | Metadata version/effective-date filter; tăng retrieval depth; query expansion |
| context_precision | Chunk không liên quan hoặc bản policy cũ đứng cao | Rerank theo source/version; loại policy superseded trước generation |
| answer_relevancy | Prompt trả lời thêm nội dung không được hỏi | Giữ nguyên intent; template trả lời từng sub-question; giới hạn độ dài |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial avg 0.6893 thấp hơn factual 0.8513, đúng kỳ vọng stress-test, nhưng vẫn cao hơn multi-hop 0.6070. Hai câu adversarial lọt bottom 10 là ID 49 (xung đột policy nghỉ phép v2023/v2024) và ID 50 (VPN cá nhân). Điều này cho thấy pipeline chưa ưu tiên ổn định policy hiện hành khi cả tài liệu cũ và mới cùng được retrieve. Cần lưu metadata version, effective_date, superseded_by và áp dụng filter trước reranking.
