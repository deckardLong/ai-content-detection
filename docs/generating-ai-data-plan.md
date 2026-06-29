# KẾ HOẠCH PHÂN BỔ DỮ LIỆU AI GENERATED (CẦN VÀ ĐỦ)

## 1. Thông số Dữ liệu Hiện tại
- **Tổng số dòng dữ liệu gốc (giả định là Human text):** 16.008 dòng.
- **Số lượng cột:** 5 cột (text, type, generator, generation_method, label).
- **Phân loại (Type):** 8 loại (wiki, fiction, news, review, social, qa, email, essay).
- **Số lượng mỗi Type:** 2.001 dòng/loại (8 x 2.001 = 16.008).
- **Mô hình AI (Generator):** 3 loại (Gemini, Claude, GPT).
- **Phương pháp sinh (Generation Method):** 3 loại (Ví dụ: Zero-shot, Few-shot, Paraphrase).

---

## 2. Tiêu chí "Cần và Đủ"
- **Cần:** Để huấn luyện một mô hình phân loại (ví dụ: phát hiện AI vs Human) hoặc phân tích hành vi tốt nhất, tỷ lệ dữ liệu giữa các lớp nên là **1:1**. Do đó, bạn cần tạo ra **chính xác 16.008 dòng văn bản do AI sinh ra** để đối ứng với 16.008 dòng gốc.
- **Đủ:** Dữ liệu AI sinh ra phải được phân phối **hoàn toàn đồng đều** (không thiên lệch/bias) giữa 8 thể loại, 3 mô hình AI và 3 phương pháp sinh.

---

## 3. Tính toán Phân bổ Chi tiết

### A. Phân bổ cho từng Thể loại (Type)
Mỗi thể loại (ví dụ: *wiki*) đang có **2.001 dòng**. Chúng ta sẽ chia đều 2.001 dòng này cho 3 AI.
- Dữ liệu mỗi AI đảm nhận trong 1 Type: `2.001 / 3 = 667 dòng`.
  *(Kiểm tra: 667 x 3 = 2.001 -> Chia hết hoàn hảo!)*

Tiếp tục chia 667 dòng của mỗi AI cho 3 Phương pháp (Method):
- Dữ liệu mỗi Method đảm nhận: `667 / 3 = 222.33 dòng`.
- Do không thể có số thập phân ở số dòng, ta phân bổ làm tròn như sau:
  - Method 1: **223 dòng**
  - Method 2: **222 dòng**
  - Method 3: **222 dòng**
  *(Kiểm tra: 223 + 222 + 222 = 667 dòng)*

### B. Bảng phân bổ cho 1 Type (VD: cho 2.001 dòng "wiki")

| Generator (AI) | Generation Method | Số lượng dòng (Cần & Đủ) |
| :--- | :--- | :---: |
| **Gemini** | Method 1 | 223 |
| | Method 2 | 222 |
| | Method 3 | 222 |
| **Claude** | Method 1 | 223 |
| | Method 2 | 222 |
| | Method 3 | 222 |
| **GPT** | Method 1 | 223 |
| | Method 2 | 222 |
| | Method 3 | 222 |
| **TỔNG CỘNG** | | **2.001 dòng** |

### C. Bảng phân bổ cho Toàn bộ Dataset (16.008 dòng)
Khi nhân kịch bản phân bổ của 1 Type lên cho 8 Types, chúng ta sẽ có con số tổng thể của toàn bộ dự án:

| Generator (AI) | Generation Method | Số lượng mỗi Type | Số lượng toàn Dataset (x8 Types) |
| :--- | :--- | :---: | :---: |
| **Gemini** | Method 1 | 223 | 1.784 |
| | Method 2 | 222 | 1.776 |
| | Method 3 | 222 | 1.776 |
| **Claude** | Method 1 | 223 | 1.784 |
| | Method 2 | 222 | 1.776 |
| | Method 3 | 222 | 1.776 |
| **GPT** | Method 1 | 223 | 1.784 |
| | Method 2 | 222 | 1.776 |
| | Method 3 | 222 | 1.776 |
| **TỔNG CỘNG**| | **2.001** | **16.008 dòng** |

---

## 4. Tóm tắt Meta-data sau khi hoàn thành
Sau khi thực hiện crawl/generate theo kế hoạch trên và gộp với dữ liệu gốc, cấu trúc dataset của bạn sẽ trông như sau:

- **Tổng kích thước:** 32.016 dòng (16.008 Gốc + 16.008 AI).
- **Cột `type`:** Mỗi loại (trong 8 loại) có chính xác 4.002 dòng.
- **Cột `generator`:** - `Human` (hoặc tên nguồn gốc): 16.008 dòng
  - `Gemini`: 5.336 dòng
  - `Claude`: 5.336 dòng
  - `GPT`: 5.336 dòng
- **Cột `generation_method`:** Tương ứng theo mapping. Đối với dữ liệu gốc, có thể để là `NaN` hoặc `human_written`.
- **Cột `label`:** - `0` (hoặc `Human`): 16.008 dòng.
  - `1` (hoặc `AI`): 16.008 dòng.

## 5. Khuyến nghị Kỹ thuật
1. **Quản lý ID (Row Mapping):** Bạn nên thêm một cột `id` hoặc `original_text_id` để biết chính xác đoạn text AI nào được sinh ra (paraphrase/dựa trên) từ đoạn text gốc nào (để phục vụ paired-evaluation nếu cần).
2. **Chi phí API:** Tính toán cẩn thận số lượng Token cho 16.008 requests. Ví dụ, nếu mỗi response khoảng 250 words, bạn sẽ tạo ra ~4 triệu words dữ liệu AI sinh ra, cần kiểm soát hạn mức (Rate Limits) của cả 3 bên API cung cấp.