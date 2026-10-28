# Các hướng cải thiện để phát triển thành paper

> Mục đích: từ project *Vietnamese AI Text Detector* (BamiBERT + Integrated Gradients + Gemini) chọn ra hướng nghiên cứu có tính mới, khả thi với dữ liệu hiện có, làm nền cho paper và khóa luận.

---

## 1. Hiện trạng và tài sản riêng của project

| Thành phần | Hiện có | Ghi chú |
|---|---|---|
| Dữ liệu | 5.000 mẫu: 2.500 human, 1.250 AI-generated, 1.250 AI-rewritten | Tin công nghệ tiếng Việt (ictnews, quantrimang, vnexpress, genk, tuoitre, thanhnien) |
| **Cặp song song** | Mỗi bài rewritten có `original_text` (bài human gốc) | **Tài sản quan trọng nhất** – biết chính xác phần nào AI giữ lại, phần nào AI thay đổi |
| Mô hình | BamiBERT fine-tune, CLS + linear, truncate 512 token | |
| Giải thích | Integrated Gradients (Captum), đánh giá bằng Infidelity / Sensitivity | |
| Giải thích bằng LLM | Gemini nhận top từ IG + thống kê câu, sinh 3–5 bullet | Có script kiểm tra groundedness (`scripts/eval_llm/`) |

### Phân tích nhanh trên tập test (`data/evaluated/test_df_evaluated.csv`)

**Độ chính xác theo loại dữ liệu**

| Loại | Accuracy | prob_ai TB | Độ dài TB (từ) |
|---|---|---|---|
| human | 97,4% | 0,04 | 1.081 |
| generated | 99,5% | 0,99 | 602 |
| rewrited | **88,7%** | 0,89 | 713 |

**Độ chính xác trên rewritten theo mức tương đồng với bài gốc** (SequenceMatcher theo từ, 400 từ đầu, chia 4 phần bằng nhau):

| Mức tương đồng | Accuracy | n |
|---|---|---|
| 0,03 – 0,21 (viết lại nhiều) | 94,1% | 51 |
| 0,21 – 0,35 | 86,3% | 51 |
| 0,35 – 0,52 | 80,0% | 50 |
| 0,52 – 0,98 (gần giữ nguyên) | 94,1% | 51 |

→ Quan hệ **hình chữ U**, không đơn điệu. Bài gần như giữ nguyên bài gốc vẫn bị nhận ra là AI 94% → cần trả lời: *mô hình đang dựa vào token nào?* Đây là động lực thực nghiệm trực tiếp cho Hướng 1.

**Khác:** tỉ lệ độ dài rewrite/gốc tương quan nghịch với `prob_ai` (Spearman ≈ −0,23) – bài rewrite càng dài hơn bản gốc càng "giống người".

### ⚠ Vấn đề cần xử lý trước khi viết paper

- **Số liệu không nhất quán:** báo cáo chương 4 ghi support 429/337 và accuracy rewritten 76%, trong khi `test_df_evaluated.csv` cho 381/385 và 88,7%. Cần chạy lại và chốt một bộ số duy nhất.
- **Tập test nhỏ** (~200 bài rewritten) → nên dùng cross-validation (theo `group_id` để tránh rò rỉ cặp gốc–rewrite) và báo khoảng tin cậy bootstrap.
- **Shortcut độ dài:** human dài gấp ~1,7 lần AI, lại truncate 512 token → cần thí nghiệm kiểm soát độ dài (cắt cùng độ dài) để chứng minh mô hình không học độ dài.

---

## 2. Hướng 1 (đề xuất chính): Dùng sai khác rewrite–gốc làm ground truth cho giải thích

### Vấn đề
Giải thích cho bộ phát hiện văn bản AI (IG, SHAP, LIME) hiện chỉ được đánh giá gián tiếp (infidelity, sensitivity, deletion curve). **Chưa có nhãn đúng ở mức token** để biết giải thích có chỉ đúng phần do AI tạo ra hay không. Tra cứu sơ bộ chưa thấy công trình nào dùng alignment giữa bài gốc và bản AI viết lại để làm ground truth cho việc đánh giá attribution của detector.

### Ý tưởng
Với mỗi cặp (gốc, rewrite), alignment cho biết token nào trong bản rewrite là:
- **copied** – giữ nguyên từ người viết,
- **AI-introduced** – do AI thêm/sửa.

Nhãn này **miễn phí, tự động, không cần gán nhãn thủ công**.

### Phương pháp
1. **Gán nhãn token:** align ở mức từ (difflib / Levenshtein), ánh xạ sang subword của BamiBERT.
2. **Thước đo đề xuất:**
   - *Attribution Precision@k*: % token AI-introduced trong top-k attribution dương.
   - *Token AUROC*: khả năng điểm attribution phân biệt AI-introduced vs copied.
   - *Attribution Mass Ratio*: tỉ lệ tổng attribution dương rơi vào token AI-introduced, chuẩn hóa theo tỉ lệ token AI-introduced (so với ngẫu nhiên).
3. **So sánh phương pháp giải thích:** IG (baseline PAD / MASK / UNK), Gradient×Input, Attention rollout, Occlusion, KernelSHAP, LIME.
4. **Phân tích theo mức độ rewrite:** giải thích tốt dần hay kém đi khi AI sửa nhiều hơn? Giải thích có lý giải được quan hệ hình chữ U ở trên không?
5. **Đóng góp phương pháp (tùy chọn nhưng làm paper mạnh hơn) – rationale supervision:** thêm loss phụ khuyến khích attribution (hoặc một token-level head) tập trung vào token AI-introduced:
   `L = L_CE + λ · L_token` (L_token: BCE trên nhãn copied/AI-introduced, chỉ áp dụng cho mẫu rewritten).
   Đo: accuracy trên rewritten, token AUROC, và độ bền khi bị paraphrase/thay từ đồng nghĩa.

### Câu hỏi nghiên cứu
- RQ1: Các phương pháp attribution phổ biến có chỉ đúng phần do AI tạo ra không?
- RQ2: Chất lượng giải thích thay đổi thế nào theo mức độ can thiệp của AI?
- RQ3: Giám sát ở mức token có cải thiện phát hiện văn bản rewritten và tính bền vững không?

### Đóng góp dự kiến
1. Một giao thức đánh giá giải thích có ground truth cho phát hiện văn bản AI (áp dụng được cho mọi ngôn ngữ có dữ liệu cặp).
2. Benchmark các phương pháp attribution trên tiếng Việt.
3. Mô hình có giám sát token cải thiện nhóm rewritten (nhóm yếu nhất).

### Rủi ro và cách giảm
| Rủi ro | Cách xử lý |
|---|---|
| Alignment nhiễu khi AI đảo trật tự câu | Align theo câu trước (sentence matching), rồi mới align từ trong câu |
| Token "copied" vẫn có thể là tín hiệu AI (AI chọn giữ lại) | Báo cáo như một giới hạn; thêm phân tích định tính |
| Ít dữ liệu | Dùng toàn bộ 1.250 cặp với k-fold; có thể sinh thêm rewrite bằng LLM khác |

---

## 3. Hướng 2 (bổ trợ): Kiểm chứng tính trung thực của giải thích do LLM sinh ra

### Vấn đề
Prompt hiện tại đưa cho Gemini các chỉ số *độ dài câu, độ lệch chuẩn, mật độ dấu câu*, nhưng **BamiBERT không dùng trực tiếp các đặc trưng này**. Gemini có thể "hợp lý hóa" kết quả bằng những lý do nghe hợp lý nhưng không phản ánh cơ chế thật của mô hình (plausible nhưng không faithful). Script groundedness hiện tại chỉ kiểm tra LLM *trích đúng số liệu*, chưa kiểm tra lời giải thích *đúng với mô hình*.

### Phương pháp
1. **Probing:** từ embedding CLS có dự đoán được các thống kê câu không? Các thống kê này có ảnh hưởng nhân quả tới `prob_ai` không (can thiệp: tách/gộp câu, thêm/bớt dấu câu nhưng giữ nội dung)?
2. **Kiểm chứng phản thực từng bullet:** mỗi bullet được chuyển thành một phép thử:
   - "Từ X đẩy về AI" → thay/xóa X → `prob_ai` phải giảm.
   - "Câu có độ dài đồng đều nên giống AI" → làm độ dài câu đa dạng hơn → `prob_ai` phải giảm.
   Bullet qua phép thử = *faithful*.
3. **Vòng lặp verify-and-regenerate:** bullet không qua kiểm chứng bị loại, gửi phản hồi cho LLM sinh lại. So sánh tỉ lệ bullet faithful trước/sau.
4. **(Tùy chọn) Đánh giá người dùng nhỏ:** 10–20 người đánh giá độ dễ hiểu và độ tin cậy giữa 3 dạng: chỉ highlight IG / Gemini gốc / Gemini đã kiểm chứng.

### Tính mới
Madsen và cộng sự (2024) kiểm tra *self-explanation* của chính LLM. Ở đây LLM giải thích **cho một mô hình khác** (detector), và lời giải thích được kiểm chứng bằng can thiệp lên chính detector – chưa thấy công trình nào làm cho bài toán phát hiện văn bản AI, đặc biệt với tiếng Việt.

### Đóng góp dự kiến
1. Chỉ ra (định lượng) mức độ LLM giải thích sai cơ chế của detector.
2. Quy trình verify-and-regenerate nâng tỉ lệ giải thích trung thực.
3. Mở rộng trực tiếp từ code đã có (`gemini_service.py`, `scripts/eval_llm/`).

---

## 4. Hướng 3 (nếu còn thời gian): Huấn luyện có kiểm soát nội dung bằng dữ liệu cặp

### Ý tưởng
Cặp (gốc, rewrite) cùng nội dung, chỉ khác văn phong → dùng để buộc mô hình học **văn phong** thay vì **chủ đề**.
- **Pair-contrastive loss:** đẩy biểu diễn của hai phần tử trong cặp ra xa dù cùng chủ đề.
- **Style direction:** ước lượng vector trung bình `h(rewrite) − h(gốc)`, chiếu loại bỏ thành phần chủ đề khỏi biểu diễn.

### Đánh giá
- Accuracy/recall trên rewritten.
- Khái quát hóa: train trên một số nguồn báo, test trên nguồn chưa thấy (leave-one-source-out); test với văn bản sinh bởi LLM chưa thấy (ví dụ chỉ train Gemini, test DeepSeek/Claude).

### Lưu ý
Contrastive learning cho phát hiện văn bản AI đã có (ví dụ DeTeCtive, NeurIPS 2024). Điểm mới chỉ nằm ở việc dùng *cặp cùng nội dung* để khử chủ đề → độ mới trung bình, nên dùng như phần mở rộng.

---

## 5. Các hướng KHÔNG nên chọn

| Hướng | Lý do |
|---|---|
| Xây dataset tiếng Việt + so sánh PhoBERT/BamiBERT/mBERT | Đã có ViDetect (UIT, 2024), cũng dùng ChatGPT rewrite bài luận (dữ liệu cặp). Điểm mới phải nằm ở **cách dùng** cặp, không phải dataset |
| Phát hiện dựa trên cho LLM viết lại rồi so sánh (Raidar, Learning-to-Rewrite) | Đã nhiều công trình tiếng Anh |
| Phân loại mức độ can thiệp AI (human / polished / rewritten / generated) | Đã có LLM-DetectAIve, APT-Eval, Beemo |
| Ước lượng mức độ dùng AI trong báo chí Việt theo thời gian | Ý tưởng hay nhưng dữ liệu hiện chỉ ~30 bài trước 2023 (2.270 bài năm 2026); cần crawl lại nhiều bài cũ |

---

## 6. Đề xuất cấu trúc khóa luận / paper

**Tên gợi ý:** *Giải thích đáng tin cậy cho phát hiện văn bản do AI tạo ra trong tiếng Việt: đánh giá dựa trên dữ liệu viết lại song song*

| Phần | Nội dung | Trạng thái |
|---|---|---|
| 1 | Bộ dữ liệu 3 loại + detector BamiBERT | Đã có (cần chốt lại số liệu) |
| 2 | Giao thức đánh giá attribution bằng ground truth từ cặp gốc–rewrite | Hướng 1 |
| 3 | Rationale supervision cải thiện nhóm rewritten | Hướng 1 (mở rộng) |
| 4 | Kiểm chứng và sinh lại giải thích của LLM | Hướng 2 |
| 5 | (Tùy chọn) Khử chủ đề bằng dữ liệu cặp | Hướng 3 |

Hướng 1 + Hướng 2 đủ cho một paper ngắn tại hội nghị trong nước (KSE, RIVF, SoICT).

### Kế hoạch thực hiện gợi ý

| Tuần | Công việc |
|---|---|
| 1 | Chạy lại đánh giá, chốt số liệu; thí nghiệm kiểm soát độ dài; k-fold theo `group_id` |
| 2–3 | Alignment + gán nhãn token; cài đặt các thước đo; benchmark các phương pháp attribution |
| 4–5 | Rationale supervision; đánh giá trên rewritten và paraphrase |
| 6–7 | Probing + kiểm chứng phản thực cho bullet Gemini; vòng verify-and-regenerate |
| 8 | Viết paper, rà soát tài liệu liên quan lần cuối |

---

## 7. Tài liệu tham khảo liên quan

- Tran et al. (2024). *Vietnamese AI Generated Text Detection* (ViDetect). arXiv:2405.03206. https://arxiv.org/abs/2405.03206
- Madsen, Chandar & Reddy (2024). *Are self-explanations from Large Language Models faithful?* Findings of ACL. https://arxiv.org/abs/2401.07927
- *Counterfactuals As a Means for Evaluating Faithfulness of Attribution Methods in Autoregressive Language Models* (2024). https://arxiv.org/abs/2408.11252
- *READER: Reasoning-Enhanced AI-Generated Text Detection*. https://www.alphaxiv.org/abs/2605.25281
- *Explainability-Based Token Replacement on LLM-Generated Text* (2025). https://arxiv.org/abs/2506.04050
- Guo et al. (2024). *DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning*. NeurIPS.
- Mao et al. (2024). *Raidar: geneRative AI Detection viA Rewriting*. ICLR.
- Abassy et al. (2024). *LLM-DetectAIve: a Tool for Fine-Grained Machine-Generated Text Detection*.
- Saha & Feizi (2025). *Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing* (APT-Eval).

> Lưu ý: việc kiểm tra tính mới mới ở mức tra cứu sơ bộ; cần rà soát thêm (Google Scholar, ACL Anthology, arXiv 2025–2026) trước khi chốt đề tài.
