# app.py

from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Flask
import matplotlib.pyplot as plt
import io
from flask_cors import CORS
import threading
import logging

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# Download NLTK stopwords
# -------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------
# Load model and tokenizer
# -------------------------
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 200

logger.info("Loading model...")
model = load_model(MODEL_PATH)
logger.info("Model loaded successfully.")

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
logger.info("Tokenizer loaded successfully.")

# -------------------------
# Preprocessing and prediction
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def encode_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded

# Thread-safe prediction
model_lock = threading.Lock()

def predict_sentiment(text):
    padded = encode_text(text)
    with model_lock:
        prob = float(model.predict(padded)[0][0])
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    return sentiment, prob

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return "Flask backend running!"

# Single review prediction
@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        review = data.get('review', '')
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        preprocessed = preprocess_text(review)
        sentiment, prob = predict_sentiment(preprocessed)
        return jsonify({'sentiment': sentiment, 'probability': prob})
    except Exception as e:
        logger.error("Error in /predict_single: %s", e)
        return jsonify({'error': str(e)}), 500

# Batch prediction from CSV
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        df = pd.read_csv(file)
        if 'review' not in df.columns:
            return jsonify({'error': 'CSV must contain a "review" column'}), 400

        preprocessed = df['review'].apply(preprocess_text)
        padded = pad_sequences(tokenizer.texts_to_sequences(preprocessed),
                               maxlen=MAX_SEQUENCE_LENGTH)

        with model_lock:
            probs = model.predict(padded).flatten()

        sentiments = ['Positive' if p >= 0.5 else 'Negative' for p in probs]
        df['sentiment'] = sentiments
        df['probability'] = probs

        csv_io = io.StringIO()
        df.to_csv(csv_io, index=False)
        csv_io.seek(0)
        return send_file(io.BytesIO(csv_io.getvalue().encode()),
                         mimetype='text/csv',
                         download_name='predictions.csv',
                         as_attachment=True)
    except Exception as e:
        logger.error("Error in /predict_batch: %s", e)
        return jsonify({'error': str(e)}), 500

# WordCloud endpoint
@app.route('/wordcloud', methods=['GET'])
def wordcloud_endpoint():
    try:
        df = pd.read_csv('dataset/imdb_train.csv')
        all_reviews = " ".join(df['review'].apply(preprocess_text).head(2000))

        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=200,
            collocations=False
        ).generate(all_reviews)

        img_io = io.BytesIO()
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.error("Error in /wordcloud: %s", e)
        return jsonify({'error': str(e)}), 500

# -------------------------
# Run Flask app
# -------------------------
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
