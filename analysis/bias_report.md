# LLM Judge Bias Report — Phase B

**Sinh viên:** Chu Phú Thành  
**Ngày:** 26/08/2026  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

Bảng dưới tóm tắt 5 trong 10 cặp model answer (A) so với trusted reference (B).
Nguồn số liệu: `reports/judge_results.json`.

| # | Question (tóm tắt) | Final winner | Reasoning tóm tắt |
|---:|---|---|---|
| 1 | Nghỉ kết hôn | B | B thêm thông tin không trừ phép năm |
| 2 | Phê duyệt thiết bị 55 triệu | B | B nêu đúng CEO, A nêu sai Director |
| 3 | Thưởng Tết từ 6 tháng | B | B đầy đủ hơn về trường hợp dưới 6 tháng |
| 4 | Senior 9 năm: phép và lương | tie | Hai answer cùng đúng các dữ kiện được hỏi |
| 5 | Hoàn trả khóa học sau 8 tháng | tie | Hai answer cùng kết luận hoàn trả 25 triệu |

---

## 2. Swap-and-Average Results

| # | Pass 1 | Pass 2 (đã quy về A/B gốc) | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | B | B | B | Yes |
| 2 | B | B | B | Yes |
| 3 | B | B | B | Yes |
| 4 | tie | tie | tie | Yes |
| 5 | tie | B | tie | No |

Trên toàn bộ 10 cặp có 2 case không nhất quán.

**Position bias rate:** **20%** (2/10). Mức này dưới ngưỡng cảnh báo 30%, nhưng đủ lớn để tiếp tục bắt buộc swap-and-average.

---

## 3. Cohen's κ Analysis

Cohen's κ dùng absolute judge so candidate answer với trusted reference. Pairwise winner không được dùng trực tiếp làm nhãn tốt/xấu, vì một answer đúng nhưng ngắn hơn reference vẫn có thể thua pairwise.

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 0 | No |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** **0.800**  
**Interpretation:** substantial agreement, sát ngưỡng almost perfect; đạt bonus κ > 0.6.

Case bất đồng ID 46 xảy ra vì judge yêu cầu thêm điều kiện trưởng phòng phê duyệt nghỉ không lương, trong khi human label coi câu trả lời hiện tại đã đủ cho câu hỏi có/không.

---

## 4. Verbosity Bias

Trong 7 case có winner rõ ràng:

- A thắng và A dài hơn B: 0/7.
- B thắng và B dài hơn A: 7/7.
- **Verbosity bias rate:** **100%** theo định nghĩa của bài.

Kết quả này cần diễn giải thận trọng: answer B trong thí nghiệm chính là trusted reference và luôn chi tiết hơn, nên độ dài bị confound với độ đầy đủ/độ chính xác. Dù vậy, reasoning cho thấy judge thường chọn B vì bổ sung chi tiết không được câu hỏi yêu cầu. Production eval nên chấm accuracy/completeness/conciseness thành ba điểm riêng và hoán đổi cả nội dung lẫn vị trí.

---

## 5. Nhận xét chung

LLM Judge có độ đồng thuận tốt với người chấm (κ 0.800), nhưng chưa nên là nguồn quyết định duy nhất. Position bias 20% chứng minh một lượt pairwise chưa đủ ổn định; swap-and-average đã biến case bất đồng thành tie thay vì quyết định sai chắc chắn. Verbosity bias đo được 100%, dù bị ảnh hưởng bởi thiết kế reference dài hơn, vẫn cho thấy rubric cần nhấn mạnh chỉ trả lời phần được hỏi. Trong production nên giữ human audit cho mẫu ngẫu nhiên, theo dõi κ theo thời gian và chặn release nếu κ dưới 0.6.
