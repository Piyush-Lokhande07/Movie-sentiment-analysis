import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("sentiment_model.h5")

df_test = pd.read_csv("./dataset/imdb_test.csv")


cleaned_reviews = [clean_text(r) for r in df_test['review']]
sequences = tokenizer.texts_to_sequences(cleaned_reviews)
padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

probs = model.predict(padded, batch_size=32)
labels = ["positive" if p >= 0.5 else "negative" for p in probs.flatten()]

df_results = pd.DataFrame({
    'review': df_test['review'],
    'predicted_sentiment': labels,
    'probability': probs.flatten()
})


if 'sentiment' in df_test.columns:
    y_true = df_test['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
    y_pred = [1 if l=='positive' else 0 for l in labels]

    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100

    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1-score:  {f1:.2f}%")
