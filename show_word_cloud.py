import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from preprocess import clean_text

df = pd.read_csv("./dataset/imdb_train.csv")

all_reviews = " ".join(df['review'].apply(clean_text))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=200
).generate(all_reviews)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

