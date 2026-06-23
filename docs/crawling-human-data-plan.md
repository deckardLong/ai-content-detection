# Kế Hoạch Thu Thập Dữ Liệu Human

## 1. Tổng Quan

| Thông tin | Chi tiết |
|---|---|
| Tổng số dòng human cần thu thập | **14,007 dòng** |
| Số loại type | **7 type** |
| Số dòng mỗi type | **2,001 dòng** |
| Ngôn ngữ | Tiếng Việt |
| Nhãn (`label`) | `0` (human) |
| `generator` | `human` |
| `generation_method` | `human` |

---

## 2. Cấu Trúc Dòng Dữ Liệu Human

```
text | type | generator | generation_method | label
"..." | essay | human | human | 0
"..." | email | human | human | 0
"..." | social | human | human | 0
"..." | fiction | human | human | 0
"..." | wiki | human | human | 0
"..." | review | human | human | 0
"..." | Q&A | human | human | 0
```

---

## 3. Phân Bổ Theo Type

| Type | Số dòng | Độ dài văn bản gợi ý |
|---|---|---|
| essay | 2,001 | 300–800 từ |
| email | 2,001 | 50–300 từ |
| social | 2,001 | 30–200 từ |
| fiction | 2,001 | 200–600 từ |
| wiki | 2,001 | 150–400 từ |
| review | 2,001 | 50–250 từ |
| Q&A | 2,001 | 100–400 từ |
| **Tổng** | **14,007** | |

---

## 4. Nguồn Thu Thập Theo Từng Type

### 4.1 Essay (`essay`) — 2,001 dòng

**Đặc điểm văn phong:** mạch lạc, có luận điểm rõ ràng, trang trọng, đoạn mở/thân/kết rõ ràng.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [VnExpress — Chuyên mục Góc nhìn](https://vnexpress.net/goc-nhin) | Bài luận quan điểm cá nhân của tác giả thật, ký tên rõ ràng | ~800 dòng |
| [Báo Tuổi Trẻ — Chuyên mục Bạn đọc](https://tuoitre.vn/ban-doc.htm) | Bài viết cảm nhận/luận điểm từ bạn đọc gửi về | ~600 dòng |
| Trang thu thập bài luận học sinh/sinh viên (các diễn đàn học thuật) | Bài luận văn, tiểu luận môn học được chia sẻ công khai | ~601 dòng |

**Lưu ý:** ưu tiên bài có tên tác giả rõ ràng, tránh bài không rõ nguồn gốc. Chỉ lấy bài được đăng **trước tháng 1/2023** để tránh rủi ro văn bản đã có AI hỗ trợ.

---

### 4.2 Email (`email`) — 2,001 dòng

**Đặc điểm văn phong:** có lời chào/kết thúc, mục đích rõ ràng, ngắn gọn, thường có danh xưng.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [ThưViệt.vn](https://thuviet.vn) / các trang mẫu thư tiếng Việt | Mẫu thư xin việc, thư cảm ơn, thư khiếu nại do người thật viết | ~700 dòng |
| Diễn đàn [Dân Trí — Hỏi đáp](https://dantri.com.vn) | Email/thư hỏi đáp bạn đọc gửi về tòa soạn | ~500 dòng |
| Kho mẫu văn bản hành chính công khai (thư hành chính, công văn ngắn) | Văn bản hành chính có người ký tên, đơn vị rõ ràng | ~801 dòng |

**Lưu ý:** email thực tế cá nhân không thể crawl vì lý do bảo mật — nên dùng các mẫu thư công khai hoặc thư được đăng tải tự nguyện trên forum.

---

### 4.3 Mạng Xã Hội / Diễn Đàn (`social`) — 2,001 dòng

**Đặc điểm văn phong:** không trang trọng, cảm xúc cá nhân, viết tắt, câu ngắn, emoji.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [VOZ Forum](https://voz.vn) | Bình luận/post của người dùng trong các thread thảo luận | ~700 dòng |
| [Reddit r/vietnam](https://reddit.com/r/vietnam) | Post tiếng Việt, bình luận của người dùng thật | ~500 dòng |
| [Tinhte.vn — Khu vực thảo luận](https://tinhte.vn) | Bình luận công nghệ, đánh giá trải nghiệm cá nhân | ~801 dòng |

**Lưu ý:** chỉ lấy các post/bình luận độc lập (không phải reply ngắn 1–2 từ như "ok", "thanks"), đảm bảo đủ độ dài tối thiểu 30 từ để model có đủ context.

---

### 4.4 Văn Học / Sáng Tác (`fiction`) — 2,001 dòng

**Đặc điểm văn phong:** tự sự, mô tả, giàu hình ảnh, có nhân vật/cảm xúc, ngôn ngữ văn chương.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [HuggingFace — ntt123/viet-tts-dataset](https://huggingface.co/datasets/ntt123/viet-tts-dataset) | Truyện ngắn/tiểu thuyết Vũ Trọng Phụng và các tác giả cùng thời, public domain | ~800 dòng |
| [Truyện ngắn Việt Nam — Wikisource](https://vi.wikisource.org) | Tác phẩm văn học Việt Nam public domain, tác giả xác định rõ | ~600 dòng |
| [Truyện ngắn trên VnExpress](https://vnexpress.net/van-hoa/van-hoc) | Truyện ngắn đăng báo, tác giả ký tên | ~601 dòng |

**Lưu ý:** ưu tiên tác phẩm **public domain** (trước 1975 hoặc đã hết hạn bản quyền) để tránh vấn đề pháp lý. Chia đoạn theo đơn vị đoạn văn (paragraph) thay vì cắt tùy tiện giữa câu.

---

### 4.5 Bách Khoa / Wikipedia-style (`wiki`) — 2,001 dòng

**Đặc điểm văn phong:** trung lập, khách quan, định nghĩa rõ ràng, không có quan điểm cá nhân.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [HuggingFace — wikimedia/wikipedia (vi)](https://huggingface.co/datasets/wikimedia/wikipedia) | Toàn bộ Wikipedia tiếng Việt, config `20231101.vi` | ~1,500 dòng |
| [Bách Khoa Toàn Thư Việt Nam](https://bachkhoatoanthu.gov.vn) | Từ điển bách khoa chính thống của Việt Nam | ~501 dòng |

**Lưu ý:** lấy phần **introduction** (đoạn đầu) của mỗi bài Wikipedia, không lấy toàn bộ bài (quá dài). Lọc bỏ các bài quá ngắn dưới 100 từ (stub articles). Đa dạng hóa chủ đề (khoa học, lịch sử, địa lý, văn hóa...) để tránh model học theo chủ đề thay vì văn phong.

---

### 4.6 Đánh Giá Sản Phẩm / Dịch Vụ (`review`) — 2,001 dòng

**Đặc điểm văn phong:** chủ quan, có cảm nhận cá nhân, so sánh, thường có ưu/nhược điểm.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [Tiki.vn — Review sản phẩm](https://tiki.vn) | Review sản phẩm của người mua thật, có tài khoản xác thực | ~800 dòng |
| [Google Maps — Review địa điểm Việt Nam](https://maps.google.com) | Đánh giá nhà hàng, khách sạn, địa điểm bằng tiếng Việt | ~700 dòng |
| [Foody.vn](https://www.foody.vn) | Đánh giá quán ăn, chi tiết về trải nghiệm cá nhân | ~501 dòng |

**Lưu ý:** lọc review có độ dài tối thiểu 50 từ (loại bỏ review chỉ 1–2 câu). Không lấy review có nội dung nghi vấn spam hoặc template lặp nhau.

---

### 4.7 Hỏi & Đáp (`Q&A`) — 2,001 dòng

**Đặc điểm văn phong:** câu hỏi rõ ràng + câu trả lời giải thích, có thể dùng ngôi thứ nhất, mang tính cá nhân hóa.

| Nguồn | Mô tả | Ước lượng |
|---|---|---|
| [Stack Overflow — Vietnamese tag](https://stackoverflow.com/questions/tagged/vietnamese) | Câu hỏi/trả lời kỹ thuật tiếng Việt | ~400 dòng |
| [VOZ — Khu vực hỏi đáp](https://voz.vn) | Câu hỏi + câu trả lời dài, giải thích chi tiết | ~800 dòng |
| [Dân Trí / VnExpress — Bạn đọc hỏi chuyên gia](https://dantri.com.vn) | Câu hỏi của bạn đọc + trả lời của chuyên gia | ~801 dòng |

**Lưu ý:** với type Q&A, mỗi **dòng dữ liệu** nên là **câu trả lời** (không phải câu hỏi) vì câu trả lời mới có đủ độ dài và đặc trưng văn phong để model học. Câu hỏi thường quá ngắn và không đủ thông tin.

---

## 5. Tiêu Chí Lọc Dữ Liệu Human (áp dụng cho tất cả type)

```
✅ Giữ lại nếu:
   - Văn bản tiếng Việt thuần (> 90% ký tự tiếng Việt)
   - Độ dài trong khoảng quy định của từng type
   - Có thể xác định tác giả là người thật
   - Được đăng/xuất bản trước tháng 1/2023

❌ Loại bỏ nếu:
   - Trùng lặp (deduplication theo cosine similarity > 0.85)
   - Quá nhiều ký tự đặc biệt / HTML tag / emoji (> 20%)
   - Văn bản dịch máy (dấu hiệu: cấu trúc cứng nhắc, lỗi ngữ pháp hệ thống)
   - Nguồn không xác định hoặc đăng sau tháng 1/2023
   - Độ dài ngoài ngưỡng quy định của type
```

---

## 6. Quy Trình Thu Thập (Pipeline)

```
Bước 1: Crawl thô
         └── Dùng BeautifulSoup / Scrapy thu thập văn bản thô
             từ các nguồn đã liệt kê ở mục 4

Bước 2: Làm sạch
         └── Loại bỏ HTML tags, ký tự đặc biệt, khoảng trắng thừa
             Chuẩn hóa encoding UTF-8

Bước 3: Lọc theo tiêu chí (mục 5)
         └── Kiểm tra độ dài, ngôn ngữ, ngày đăng, deduplication

Bước 4: Gán metadata
         └── Điền cột: type / generator = "human" /
             generation_method = "human" / label = 0

Bước 5: Kiểm tra thủ công mẫu ngẫu nhiên
         └── Sample 50 dòng/type → đọc thủ công để
             đảm bảo chất lượng trước khi dùng
```

---

## 7. Rủi Ro Cần Lưu Ý

| Rủi ro | Mức độ | Cách xử lý |
|---|---|---|
| Văn bản human bị nhiễm AI (người viết có AI hỗ trợ) | Cao (đặc biệt sau 2023) | Chỉ lấy văn bản trước tháng 1/2023 |
| Review spam / nội dung template | Trung bình | Lọc bằng deduplication + độ dài tối thiểu |
| Mất cân bằng chủ đề trong cùng 1 type | Trung bình | Đa dạng hóa nguồn crawl, không chỉ dùng 1 trang |
| Vi phạm ToS của trang web khi crawl | Cao | Kiểm tra robots.txt, dùng public API nếu có, crawl chậm (delay 1–2s/request) |
| Bản quyền nội dung | Trung bình | Chỉ dùng cho mục đích nghiên cứu, ưu tiên nguồn open license |