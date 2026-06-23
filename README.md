# Vietnamese Sentiment Analysis
Applying NLP techniques to the VLSP dataset.

**Course:** Modern Approaches to Natural Language Processing  
**Instructor:** Assoc. Prof. Dr. Quan Thanh Tho  
**Student:** Nguyen Quoc Bao (Student ID: 2470076)  

---

## 📝 Project Overview
This project focuses on the task of Vietnamese text sentiment analysis. It implements and compares traditional deep learning architectures alongside modern Large Language Models (Pre-trained Language Models) to evaluate performance benchmarks against computational costs.

### Implemented Models:
1. **TextCNN**
2. **BiLSTM - CNN**
3. **BiLSTM - CNN combined with Multi-Head Attention**
4. **PhoBERT** (State-of-the-art pre-trained model optimized for Vietnamese)
5. **XLM-RoBERTa**

---

## 📂 Repository Structure
* `docs/`: Contains the detailed project report in Word format (`2470076_NguyenQuocBao.docx`).
* `notebooks/`: Contains the Jupyter Notebook covering data preprocessing, model training, and evaluation (`2470076_NguyenQuocBao.ipynb`).

---

## 🚀 Getting Started

### 1. Data Preparation
Download the required dataset files and upload them to your personal Google Drive using the following path: `MyDrive/NLP/Assignment`.
* Reference Data Link (if applicable): [Google Drive Folder](https://drive.google.com/drive/folders/179OY3vLY7P4RzeCqDwBnutK-oBu-dwQ0?usp=sharing)

### 2. Execution Environment
* It is highly recommended to run the notebook via **Google Colab**.
* Enable hardware acceleration by switching the runtime to a **GPU (preferably T4 or L4)** to significantly accelerate training times, especially for PhoBERT and XLM-RoBERTa.
* If computational resources are limited, consider evaluating only the XLM-RoBERTa model or reducing the training batch size.

---

## 📊 Summary of Results & Remarks
* **Traditional Models (TextCNN, BiLSTM):** Maintained a stable accuracy rate of approximately ~70% while requiring minimal computational overhead.
* **Pre-trained Transformer Models (PhoBERT, XLM-RoBERTa):** Demonstrated substantial improvements in contextual semantic understanding, though they demand higher hardware resource allocation.
* For an in-depth analysis of text features, preprocessing workflows (Data Cleaning, Tokenization, Vectorization), and comparative benchmark metrics (mIoU, Accuracy, F1-Score), please refer to the comprehensive report located in the `docs/` directory.
