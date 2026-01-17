import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def build_recommender(df):
    df['combined_text'] = df['Job Title'] + ' ' + df['Job Description']
    df['clean_text'] = df['combined_text'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    return tfidf, tfidf_matrix

def recommend_jobs(user_input, df, tfidf, tfidf_matrix, top_n=5):
    user_input = clean_text(user_input)
    user_vec = tfidf.transform([user_input])

    similarity = cosine_similarity(user_vec, tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]

    return df.iloc[top_indices][['Job Title', 'Job Description']]
