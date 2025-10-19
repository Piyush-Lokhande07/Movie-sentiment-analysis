import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("sentiment_model.h5")


def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    prob = model.predict(padded)[0][0]
    label = "positive" if prob >= 0.5 else "negative"
    return label, prob


print("Enter 'exit' to quit.")
while True:
    review = input("\nEnter a movie review: ")
    if review.lower() == 'exit':
        break
    label, prob = predict_sentiment(review)
    print(f"Predicted Sentiment: {label}")
    print(f"Probability: {prob*100:.2f}%")
