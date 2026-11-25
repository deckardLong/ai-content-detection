# So sánh PhoBERT và BamiBERT cho phát hiện văn bản tiếng Việt do AI tạo

Thư mục này chứa toàn bộ code thí nghiệm của bài báo: hai encoder đóng băng (PhoBERT, BamiBERT)
kết hợp với các head học sâu (CNN, BiLSTM, và head kết hợp CNN‖BiLSTM), đánh giá bằng
5-fold StratifiedGroupKFold.

| Mô hình | Script huấn luyện |
|---|---|
| PhoBERT + CNN, PhoBERT + BiLSTM, BamiBERT + CNN, BamiBERT + BiLSTM | `run_all.py` |
| PhoBERT + CNN‖BiLSTM, BamiBERT + CNN‖BiLSTM | `cv_hybrid.py` |

---

## 0. Chuẩn bị

**Môi trường** (Windows, PowerShell, từ thư mục gốc của repo):

```powershell
python -m venv ..\venv
..\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Cần GPU NVIDIA có CUDA (đã chạy trên RTX 2070, PyTorch 2.7.1 + CUDA 11.8).
Nếu không có GPU, code vẫn chạy trên CPU nhưng chậm hơn nhiều.

**Dữ liệu:** đặt file `data/it_ai_human_data.csv` ở thư mục gốc của repo
(đường dẫn khai báo trong `configs/experiment.yaml`, mục `data_path`).
Bộ dữ liệu có trên Hugging Face:
<https://huggingface.co/datasets/DeckardLong/ai-content-detection-dataset>

**Cấu hình:** mọi siêu tham số nằm trong [configs/experiment.yaml](configs/experiment.yaml):
seed, số fold, số âm tiết giữ lại (225), độ dài token tối đa, batch size, learning rate, số epoch,
và cấu hình từng head. Không cần sửa gì để chạy lại kết quả của bài.

> Mọi lệnh bên dưới chạy **từ thư mục gốc của repo**. Script tự chuyển vào `comparison/`,
> nên đường dẫn đầu ra luôn là `comparison/results/...` và `comparison/models/...`.

---

## 1. Kiểm tra nhanh (smoke test, khoảng 2 phút)

Chạy thử trên 200 mẫu, 2 fold, 2 epoch để chắc chắn môi trường và dữ liệu ổn:

```powershell
python comparison/scripts/run_all.py --limit 200 --epochs 2 --folds 2
```

Kết quả nằm trong `comparison/results/frozen_smoke/` và không ảnh hưởng đến lần chạy thật.

---

## 2. (Tuỳ chọn) Chọn số âm tiết giữ lại

```powershell
python comparison/scripts/choose_truncation.py
```

In ra, với mỗi số âm tiết N, tỉ lệ bài vừa 256 token của PhoBERT và số token BamiBERT cần.
Bài báo dùng N = 225 (99.58% bài vừa PhoBERT). Bước này chỉ cần nếu bạn muốn chọn lại N.

---

## 3. Bốn mô hình chính (khoảng 1 giờ)

```powershell
python comparison/scripts/run_all.py
```

- Chia 5 fold (lưu ở `folds.json`), trích đặc trưng mỗi encoder **một lần** (cache ở
  `results/cache/`), rồi huấn luyện 4 mô hình × 5 fold.
- Thời gian tham khảo trên RTX 2070: trích đặc trưng 41 s (PhoBERT) và 105 s (BamiBERT);
  huấn luyện 95 đến 206 s mỗi fold tùy mô hình.
- **Chạy tiếp khi bị ngắt:** chạy lại đúng lệnh trên. Fold đã xong được bỏ qua, fold đang dở
  tiếp tục từ epoch cuối đã lưu. Muốn chạy lại từ đầu thì xoá `results/frozen_equal_text/`.

Đầu ra trong `comparison/results/frozen_equal_text/`:

| File | Nội dung |
|---|---|
| `folds.json` | Chỉ số train/val/test của từng fold (dùng chung cho mọi mô hình) |
| `preds/<model>_fold<k>.npz` | Dự đoán, xác suất, lịch sử loss và thời gian của từng fold |
| `metrics.json` | Trung bình ± độ lệch chuẩn qua 5 fold, kèm thời gian |

---

## 4. Head kết hợp CNN‖BiLSTM (khoảng 20 đến 30 phút mỗi encoder)

Chạy **sau** bước 3, vì script dùng lại `folds.json` để chia fold giống hệt bốn mô hình chính.

```powershell
python comparison/scripts/cv_hybrid.py --encoder bamibert
python comparison/scripts/cv_hybrid.py --encoder phobert
```

Đầu ra:

| File | Nội dung |
|---|---|
| `preds_hybrid/<encoder>+cnn_bilstm_fold<k>.npz` | Dự đoán từng fold |
| `hybrid_metrics.json` (BamiBERT), `hybrid_metrics_phobert.json` (PhoBERT) | Kết quả 5-fold |

**Đo lại thời gian huấn luyện** (không ghi đè kết quả trên):

```powershell
python comparison/scripts/cv_hybrid.py --encoder bamibert --retime
python comparison/scripts/cv_hybrid.py --encoder phobert --retime
```

Ghi vào `hybrid_timing_<encoder>.json`. Chạy **lần lượt**, không chạy song song, và để máy rảnh
(đóng trình duyệt, game, LaTeX build tự động) thì thời gian đo mới so sánh được.

---

## 5. Xuất bảng kết quả

```powershell
python comparison/scripts/export_table.py           # main_table.* và time_table.*
python comparison/scripts/compare_significance.py   # significance.csv (McNemar)
python comparison/scripts/export_extra_tables.py    # type_table.* (độ chính xác theo loại văn bản)
```

Bảng ở dạng `.md`, `.csv` và `.tex` trong `results/frozen_equal_text/`.
`compare_models.py` in nhanh bảng kết quả ra màn hình.

> Hai mô hình hybrid **không** nằm trong các bảng tự động này. Số của chúng lấy từ
> `hybrid_metrics*.json` và `hybrid_timing_*.json`.

---

## 6. Vẽ hình

```powershell
# Đường cong loss (6 mô hình, gồm cả hybrid nét đứt)
python comparison/scripts/plot_losses.py --source cv --lang en
python comparison/scripts/plot_losses.py --source cv --lang vi

# Hình phân tích: phân bố P(AI), t-SNE, lỗi chỉ một encoder đúng
python comparison/scripts/plot_analysis.py --lang en
python comparison/scripts/plot_analysis.py --lang vi
```

`--only scores|tsne|overlap|paper` để vẽ riêng một hình. Hình nằm trong `results/frozen_equal_text/plots/`.

Hình dùng trong bài báo (`_en` cho bản tiếng Anh, không hậu tố cho bản tiếng Việt):

| Trong bài | File sinh ra |
|---|---|
| `fig_loss_cv.png` | `loss_cv_all_en.png` / `loss_cv_all.png` |
| `fig_score_distribution.png` | `paper_score_distribution_v_en.png` / `paper_score_distribution_v.png` |
| `fig_error_overlap.png` | `paper_error_overlap_en.png` / `paper_error_overlap.png` |

---

## 7. (Tuỳ chọn) Mô hình dùng được ngay để dự đoán

Các bước 3 và 4 chỉ để đánh giá, không lưu mô hình. Để có file mô hình dùng cho dự đoán,
huấn luyện trên một lần chia 70/15/15 (theo nhóm):

```powershell
python comparison/scripts/train.py --all           # 4 mô hình chính
python comparison/scripts/train_hybrid.py          # BamiBERT + CNN‖BiLSTM
python comparison/scripts/evaluate.py --all        # đánh giá trên tập test 766 mẫu
python comparison/scripts/plot_losses.py           # loss của lần chia này
```

Mô hình lưu ở `comparison/models/frozen_equal_text/<encoder>+<head>.pt`.

**Dự đoán văn bản mới** (mặc định BamiBERT + CNN‖BiLSTM):

```powershell
python comparison/scripts/predict.py --text "Nội dung bài viết ..."
python comparison/scripts/predict.py --file bai_viet.txt
python comparison/scripts/predict.py --csv data.csv --column text --out ket_qua.csv
python comparison/scripts/predict.py --text "..." --heads cnn bilstm cnn_bilstm   # so sánh các head
```

Kết quả là P(AI) của mỗi head; P(AI) ≥ 0.5 nghĩa là văn bản được xếp vào AI.

---

## Thứ tự chạy tóm tắt

```powershell
python comparison/scripts/run_all.py --limit 200 --epochs 2 --folds 2   # 1. kiểm tra
python comparison/scripts/run_all.py                                    # 3. 4 mô hình
python comparison/scripts/cv_hybrid.py --encoder bamibert               # 4. hybrid
python comparison/scripts/cv_hybrid.py --encoder phobert
python comparison/scripts/export_table.py                               # 5. bảng
python comparison/scripts/compare_significance.py
python comparison/scripts/export_extra_tables.py
python comparison/scripts/plot_losses.py --source cv --lang en          # 6. hình
python comparison/scripts/plot_analysis.py --lang en
```

## Cấu trúc thư mục

```
comparison/
├── configs/experiment.yaml   # toàn bộ siêu tham số
├── scripts/                  # các lệnh chạy ở trên
├── src/
│   ├── data/                 # đọc dữ liệu, cắt 225 âm tiết, tách từ, chia fold, cache đặc trưng
│   ├── models/               # encoder, head CNN / BiLSTM / CNN‖BiLSTM, mô hình phân loại
│   ├── training/             # vòng huấn luyện, seed
│   └── evaluation/           # metric, McNemar
├── results/                  # kết quả đánh giá chéo, bảng, hình, cache đặc trưng
└── models/                   # file mô hình từ train.py / train_hybrid.py
```
