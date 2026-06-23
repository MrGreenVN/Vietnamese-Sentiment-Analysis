# Vietnamese-Sentiment-Analysis
Apply NLP for VLSP dataset
# Phân Loại Cảm Xúc Văn Bản Tiếng Việt (Vietnamese Sentiment Analysis)

Học phần: **Cách tiếp cận hiện đại trong Xử lý ngôn ngữ tự nhiên** Giảng viên hướng dẫn: **PGS.TS. Quản Thành Thơ** Sinh viên thực hiện: **Nguyễn Quốc Bảo** (MSSV: 2470076)

---

## 📝 Giới thiệu Dự án
Dự án tập trung vào bài toán phân loại cảm xúc văn bản tiếng Việt sử dụng cả các mô hình học sâu truyền thống và các mô hình ngôn ngữ lớn (Pre-trained Language Models) hiện đại nhằm so sánh, đánh giá hiệu năng và chi phí tính toán.

### Các mô hình được triển khai trong dự án:
1. **TextCNN**
2. **BiLSTM - CNN**
3. **BiLSTM - CNN kết hợp Multi-Head Attention**
4. **PhoBERT** (Pre-trained dành riêng cho tiếng Việt)
5. **XLM-RoBERTa**

---

## 📂 Cấu trúc Kho lưu trữ
* `docs/`: Chứa file báo cáo chi tiết dạng Word (`2470076_NguyenQuocBao.docx`).
* `notebooks/`: Chứa file mã nguồn triển khai toàn bộ quá trình tiền xử lý, huấn luyện và đánh giá (`2470076_NguyenQuocBao.ipynb`).

---

## 🚀 Hướng dẫn Chạy Code

### 1. Chuẩn bị dữ liệu:
Tải các file dữ liệu cần thiết và tải lên Google Drive cá nhân của bạn theo đường dẫn: `MyDrive/NLP/Assignment`.
* Link dữ liệu tham khảo (nếu có): [Google Drive Folder](https://drive.google.com/drive/folders/179OY3vLY7P4RzeCqDwBnutK-oBu-dwQ0?usp=sharing)

### 2. Môi trường thực thi:
* Khuyến khích chạy file notebook trên **Google Colab**.
* Nên cấu hình Hardware accelerator sang **GPU (ưu tiên L4 hoặc T4)** để tăng tốc độ huấn luyện mô hình (đặc biệt là PhoBERT và XLM-RoBERTa).
* Nếu tài nguyên tính toán hạn chế, bạn có thể cân nhắc chỉ chạy kiểm thử mô hình XLM-RoBERTa hoặc giảm dung lượng batch size.

---

## 📊 Kết quả và Nhận xét (Tóm tắt)
* Các mô hình truyền thống (TextCNN, BiLSTM) giữ mức độ chính xác ổn định quanh mức ~70% với chi phí tính toán thấp.
* Các mô hình Transformer pre-trained (PhoBERT, XLM-RoBERTa) đem lại cải thiện rõ rệt về khả năng hiểu ngữ nghĩa, tuy nhiên yêu cầu tài nguyên phần cứng lớn hơn.
* Chi tiết phân tích đặc trưng văn bản, các bước tiền xử lý (Clean data, Tokenization, Vectorization) và bảng biểu so sánh (mIoU, Accuracy, F1-Score) vui lòng xem tại báo cáo trong thư mục `docs/`.
