
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import re

# Constants
DATA_PATH = 'data/movies_sample.csv'
MODELS_DIR = 'models/'
MIN_VOTE_COUNT = 50
MAX_FEATURES = 5000

def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """Loads and filters the dataset by minimum vote count."""
    print("Loading data...")
    cols = ['id', 'title', 'genres', 'overview', 'keywords', 'credits', 'vote_count', 'vote_average', 'poster_path']
    df = pd.read_csv(filepath, usecols=cols)
    print(f"Initial row count: {len(df)}")
    
    df_filtered = df[df['vote_count'] >= MIN_VOTE_COUNT].copy()
    print(f"Row count after vote filter (>={MIN_VOTE_COUNT}): {len(df_filtered)}")
    
    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered

def clean_text_column(text) -> str:
    """Cleans text columns by refreshing separators."""
    if pd.isna(text):
        return ''
    return str(text).replace('-', ' ')

def create_soup(df: pd.DataFrame) -> pd.DataFrame:
    """Combines metadata features into a single string for vectorization."""
    print("Creating metadata 'soup'...")
    df['genres'] = df['genres'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    df['credits'] = df['credits'].fillna('')
    df['overview'] = df['overview'].fillna('')
    
    df['genres_str'] = df['genres'].apply(clean_text_column)
    df['keywords_str'] = df['keywords'].apply(clean_text_column)
    df['credits_str'] = df['credits'].apply(clean_text_column)
    
    df['soup'] = (
        df['title'] + ' ' + 
        df['genres_str'] + ' ' + 
        df['keywords_str'] + ' ' + 
        df['credits_str'] + ' ' + 
        df['overview']
    )
    return df

def train():
    """Executes the training pipeline and saves artifacts."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    df = load_and_filter_data(DATA_PATH)
    df = create_soup(df)
    
    print("Vectorizing data (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
    print("Training model (NearestNeighbors)...")
    nn_model = NearestNeighbors(n_neighbors=21, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(tfidf_matrix)
    
    print("Saving artifacts...")
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(nn_model, os.path.join(MODELS_DIR, 'nn_model.joblib'))
    joblib.dump(tfidf_matrix, os.path.join(MODELS_DIR, 'tfidf_matrix.joblib'))
    
    df_meta = df[['id', 'title', 'genres', 'poster_path', 'vote_average', 'overview', 'vote_count']]
    joblib.dump(df_meta, os.path.join(MODELS_DIR, 'movies_metadata.joblib'))
    
    indices = pd.Series(df_meta.index, index=df_meta['title']).drop_duplicates()
    joblib.dump(indices, os.path.join(MODELS_DIR, 'indices.joblib'))
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train()
