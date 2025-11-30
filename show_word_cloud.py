# show_word_cloud.py
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from preprocess import clean_text

# -------------------------
# Load dataset
# -------------------------
try:
    df = pd.read_csv("./dataset/imdb_train.csv")
except FileNotFoundError:
    print("Error: Dataset not found at './dataset/imdb_train.csv'")
    exit(1)

# -------------------------
# Preprocess and combine reviews
# Limit to first 2000 reviews to prevent MemoryError
# -------------------------
all_reviews = " ".join(df['review'].apply(clean_text).head(2000))

# -------------------------
# Generate WordCloud
# -------------------------
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=200,       # limit number of words
    collocations=False   # disable bigram collocations to save memory
).generate(all_reviews)

# -------------------------
# Display WordCloud
# -------------------------
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
