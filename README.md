
# IMDB Movie Review Sentiment Analysis

## **Project Overview**

This project performs **sentiment analysis** on IMDb movie reviews. It classifies reviews as **positive** or **negative** using a **Bi-directional LSTM (BiLSTM) deep learning model**.

The goal is to demonstrate NLP preprocessing, model training, and deployment in Python, with both **batch and interactive prediction modes**, along with visualizing word frequency using a **WordCloud**.

---

## **Tech Stack**

* **Programming Language:** Python 3.13.7
* **Libraries / Frameworks:**

  * **TensorFlow / Keras** → Deep learning model (BiLSTM)
  * **NLTK** → Text preprocessing, stopwords removal
  * **Pandas** → Data loading and manipulation
  * **NumPy** → Numerical computations
  * **Scikit-learn** → Metrics: accuracy, precision, recall, F1-score
  * **Matplotlib** → Plotting WordCloud
  * **WordCloud** → Visualization of most frequent words

---

## **Dataset**

* Original dataset: 50,000 IMDb reviews from Kaggle with columns:

  * `review` → text of the movie review
  * `sentiment` → `positive` or `negative` label

* For training efficiency, the dataset was reduced to **20,000 rows**.

* **Split for project**:

  * `dataset/imdb_train.csv` → 16,000 rows (training)
  * `dataset/imdb_test.csv` → 4,000 rows (testing)

---

## **Preprocessing**

* Convert all text to lowercase
* Remove punctuation and special characters
* Remove stopwords (common words like “the”, “is”, etc.)
* Tokenize text using `Tokenizer` from Keras
* Pad sequences to a fixed length of 200 for BiLSTM input

**Preprocessing code is in:** `preprocess.py`

---

## **Model**

* **Architecture**: Bi-directional LSTM (2 layers) + Dense layers + Dropout
* Input: tokenized and padded sequences
* Output: probability of being **positive**
* Trained using **TensorFlow / Keras**
* Saved as:

  * `sentiment_model.h5` → trained BiLSTM model
  * `tokenizer.pkl` → fitted tokenizer

---

## **Running Locally**

1. Clone the repository:

   ```bash
   git clone https://github.com/Piyush-Lokhande07/Movie-sentiment-analysis.git
   cd Movie-sentiment-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Batch predictions** on `imdb_test.csv`:

   ```bash
   python3 predict_batch.py
   ```

   * Saves predictions with probability in `test_predictions.csv`
   * Prints **accuracy, precision, recall, F1-score (%)** if labels are available

4. **Interactive single review prediction**:

   ```bash
   python3 predict_single.py
   ```

   * Enter a review and see predicted sentiment with probability
   * Type `exit` to quit

5. **Generate WordCloud**:

   ```bash
   python3 show_word_cloud.py
   ```

---

## **Folder Structure**

```
Movie-sentiment-analysis/
 ┣ 📄 preprocess.py
 ┣ 📄 predict_batch.py
 ┣ 📄 predict_single.py
 ┣ 📄 show_word_cloud.py
 ┣ 📄 sentiment_model.h5
 ┣ 📄 tokenizer.pkl
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┗ 📁 dataset/
     ┣ 📄 imdb_train.csv
     ┗ 📄 imdb_test.csv
```

