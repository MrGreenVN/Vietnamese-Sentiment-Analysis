# Vietnamese Sentiment Analysis

**Course:** Modern Approaches to Natural Language Processing  
**Instructor:** Assoc. Prof. Dr. Quan Thanh Tho  
**Student:** Nguyen Quoc Bao (Student ID: 2470076)  
**Institution:** Ho Chi Minh City University of Technology - VNU-HCM, Master's Program 2026

---

## I. Dataset Description and Analysis

### 1. Data Structure
Each sample in the dataset (VLSP Sentiment) consists of two main components:
* **Class (Sentiment Label):**
  * `-1`: Negative
  * `0`: Neutral
  * `1`: Positive
* **Data:** The text content of opinions/comments in Vietnamese.

The dataset is split into two subsets with a perfect class balance:
* **Training Set (Train):** 5,100 samples (1,700 samples/class).
* **Testing Set (Test):** 1,050 samples (350 samples/class).

> This balanced distribution ensures that the models learn the characteristics of all three classes fairly, preventing bias and guaranteeing that the evaluation metrics truly reflect the actual classification capability of the models.

### 2. Exploratory Data Analysis (EDA)
* **Average Length by Label:** The Neutral (0) label has the highest average length (173.5 characters), which is significantly longer than Negative (112.2) and Positive (103.6). Neutral comments tend to be detailed and descriptive, whereas positive/negative comments are usually more direct and concise.
* **Boxplot of Lengths:** All three labels show a distinct right-skewed distribution. The median lies between 70–100 characters, indicating that most actual samples are quite short, though there are multiple extremely long outliers at the upper end.
* **Train vs. Test Comparison:** The Q1, median, and Q3 percentiles between the Train and Test sets are almost identical (81 vs 81 at the median), proving that the Test set is highly representative.

### 3. Noise Analysis
An analysis of the 6,150 total samples reveals a high level of noise, which is typical for social media data:
* **Code-switching (English - Vietnamese):** Appears in **85%** of all samples (Technical terms, brand names like: *"ngốn pin"* - battery-draining, *"độ trễ bluetooth"* - bluetooth latency). The Neutral label has the highest rate of language mixing (**91.5%**).
* **Repeated Words and Characters:** Accounts for **12.7%** of all samples (e.g., *"chán lắm....."* - so boring, *"bực mình!!!!"* - frustrating!!!!). This type of noise poses a major challenge for the Tokenizer.
* **Other Noise:** Emojis or special characters account for **9.9%**, and extra whitespaces make up **0.8%**.

---

## II. Data Preprocessing

The text preprocessing pipeline standardizes the input data through the following steps:

1. **Data Cleaning:**
   * Convert all text to lowercase and normalize Unicode characters to the **NFC** form.
   * Shorten repeated expressive characters (e.g., *"đẹppppp"* $\rightarrow$ *"đẹp"*, *"!!!"* $\rightarrow$ *'!"*).
   * Normalize teencode and common abbreviations using a manually built mapping dictionary (e.g., *"ko/k"* $\rightarrow$ *"không"*, *"dc"* $\rightarrow$ *"được"*, *"ship"* $\rightarrow$ *"giao_hàng"*).
   * Use Regular Expressions (Regex) to remove non-semantic special characters, keeping only letters, numbers, and basic punctuation marks.
   * Concatenate negative phrases and modifiers into compound words (e.g., *"không tốt"* $\rightarrow$ *"không_tốt"*, *"rất hài lòng"* $\rightarrow$ *"rất_hài_lòng"*).

2. **Tokenization:**
   * Utilize a specialized Vietnamese word segmentation library (`pyvi`) to handle multi-syllable compound words (e.g., *"sản phẩm"* $\rightarrow$ *"sản_phẩm"*).
   * Cross-reference tokens with a **Word2Vec** embedding dictionary. If a compound word is missing, the system breaks it down into individual single words to prevent information loss.

3. **Vectorization:**
   * Fit a Tokenizer to map vocabulary tokens to unique integer indices (trained strictly on the Train set to avoid **Data Leakage**). Words outside the vocabulary are assigned an `OOV` token.
   * Apply **Padding** to standardize all input sequences to a fixed length.
   * Transform the target labels (`Class`) into binary vectors using **One-hot Encoding**.

---

## III. Model Architectures

Five different model architectures were benchmarked, ranging from traditional deep learning to advanced Transformer models:

### 1. TextCNN
* Initializes the embedding layer using static, pre-trained **Word2Vec** weight matrices (non-trainable).
* Uses `SpatialDropout1D` to mitigate overfitting.
* Applies parallel 1D Convolutional layers (`Conv1D`) with various kernel sizes (2, 3, 4) combined with **Dilation** techniques to expand the receptive field.
* Pools features via `GlobalMaxPooling1D`, passes them through a Dense layer (with ReLU activation), and applies `BatchNormalization` before reaching the output classification layer.

### 2. BiLSTM - CNN Model
* Shares a similar setup to TextCNN, but the embedding layer is fine-tuned during training (trainable = True).
* Sequences are passed through a **Bidirectional LSTM (BiLSTM)** layer to capture long-range contextual dependencies from both directions (left-to-right and right-to-left).
* The sequential output of the BiLSTM is then fed into parallel `Conv1D` layers to extract local n-gram features.

### 3. BiLSTM - CNN with Multi-Head Attention
* Enhanced by integrating a **Multi-Head Attention** mechanism (4 heads) directly after the BiLSTM layer (64 hidden units).
* The Attention output is combined with the BiLSTM output via a **Residual Connection** followed by `LayerNormalization`.
* Feature extraction is split into two streams: Stream 1 uses `GlobalMaxPooling1D` for global sentence representation; Stream 2 uses parallel `Conv1D` blocks for local features.
* The model uses `Categorical Crossentropy` loss with **Label Smoothing** to prevent overconfident predictions.

### 4. PhoBERT Model
* Utilizes the pre-trained **PhoBERT-base** encoder architecture, which is specifically optimized for the Vietnamese language.
* Inputs are prepared with special `[CLS]` and `[SEP]` tokens. Deep contextual representations are learned via Transformer Encoders featuring Self-Attention and Feed-Forward networks.
* The vector corresponding to the `[CLS]` token is used as the sentence representation and is passed to a Classification Head, optimized via `AdamW` with a low learning rate.

### 5. XLM-RoBERTa Model
* Evaluates a massive, state-of-the-art **Multilingual** Transformer architecture.
* Leverages global Self-Attention to test whether a multilingual model can outperform a monolingual Vietnamese model when dealing with text containing heavy code-switching and borrowed terms (like the VLSP dataset).

---

## IV. Training, Evaluation, and Analysis

### 1. Training Results and General Evaluation

| Model | Parameters | Training Time | Test Acc |
| :--- | :---: | :---: | :---: |
| **TextCNN** | 3,530,787 | 21.59s | 0.6475 |
| **BiLSTM - CNN** | 3,880,227 | 23.44s | **0.7041** |
| **LSTM - Attention** | 3,417,443 | 34.02s | 0.6686 |
| **PhoBERT** | 135,000,579 | 217.37s | **0.7562** |
| **XLM-RoBERTa** | 278,045,955 | 341.93s | 0.7162 |

#### Analysis:
* **Accuracy:** **PhoBERT** achieved the highest accuracy (**0.7562**), demonstrating the clear advantage of domain-specific pre-training on large-scale Vietnamese corpora.
* **Performance-Cost Trade-off:** Transformer models demand heavy computational resources, leading to longer training times and larger parameter sizes. Conversely, traditional models like **BiLSTM-CNN** offer exceptional economic efficiency—reaching over **70%** accuracy in just **23.44 seconds**.
* Despite its massive size (278M+ parameters), the multilingual **XLM-RoBERTa** underperformed compared to PhoBERT because it lacks language-specific optimization for Vietnamese grammar and syntax.

### 2. Error Analysis

Model interpretability methods (such as Attention or LIME) were used to extract word attribution scores (positive values support the predicted class, while negative values push the prediction toward other classes). The analysis revealed systematic weaknesses:

* **Case 1: Sentences with Contrastive Structures (The conjunction "Nhưng" - But)**
  * *Sample:* *"Đang dùng mx1. Cũng ngon nhưng chưa đầy năm mà đã 3 lần tháo ra thay 3 nút bấm rồi..."* (Actual: **Negative**).
  * *Error:* TextCNN, BiLSTM-CNN, and LSTM-Attention all misclassified this as **Positive**. They were heavily biased by positive keywords like *"ngon"* (good) and *"may"* (luckily), ignoring the sentiment shift introduced by *"nhưng"* (but). PhoBERT predicted **Neutral** because it recognized both sides but failed to give dominant weight to the negative clause.
* **Case 2: Long Negative Structures**
  * *Sample:* *"ĐÚNG LÀ MUA Ở VIỆT NAM KHÔNG ỨNG DỤNG ĐƯỢC GÌ NHIỀU..."* (Actual: **Negative**).
  * *Error:* The shallower networks misclassified this as **Positive** due to high positive weights assigned to neutral surface words like *"NAM"*, *"MUA"* (buy), and *"VIỆT"*, failing to capture the long-range dependency of the negation *"KHÔNG... NHIỀU"* (not... much). Only **PhoBERT** correctly classified this case, thanks to its deep contextual encoding.

---

## V. Conclusion and Future Directions

### Conclusion
1. This project successfully implemented and evaluated multiple generations of NLP architectures on a Vietnamese Sentiment Analysis task.
2. The **PhoBERT** language model set the benchmark for accuracy (0.7562). However, traditional architectures like **BiLSTM-CNN** still hold significant practical value (Acc > 70% while being 10x faster), making them ideal for hardware-constrained systems.
3. The biggest remaining hurdle is the model's capacity to interpret complex semantic structures, such as sarcasm, long-range negations, contrastive clauses, and colloquial language mix-ins.

### Future Work
* **Data Augmentation:** Focus on gathering and generating challenging training samples containing complex contrastive clauses to robustly train the models.
* **Architecture Tuning:** Investigate integrating sentiment lexicons into the deep learning embedding layers as supplementary features.
* **Optimization:** Experiment with Knowledge Distillation methods to transfer insights from heavy models like PhoBERT to compact CNN/RNN architectures, balancing accuracy and computational cost.

---

## References

```text
[1] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016.
[2] A. Krizhevsky, V. Nair, and G. Hinton, "Learning multiple layers of features from tiny images," University of Toronto, Tech. Rep., 2009.
[3] A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems, 2017, vol. 30.
[4] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.
[5] D. Q. Nguyen and A. T. Nguyen, “PhoBERT: Pre-trained language models for Vietnamese,” in Findings of the Association for Computational Linguistics (EMNLP), 2020.
