
import os
import joblib
import pandas as pd
import difflib
from typing import List, Dict, Union, Any

class MovieRecommender:
    """Handles movie recommendations using a pre-trained NearestNeighbors model."""
    
    def __init__(self, models_dir: str = 'models/'):
        self.models_dir = models_dir
        self.tfidf = None
        self.nn_model = None
        self.tfidf_matrix = None
        self.movies_df = None
        self.indices = None
        
        self.load_models()

    def load_models(self):
        """Loads model artifacts from disk."""
        print("Loading models...")
        try:
            self.tfidf = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.joblib'))
            self.nn_model = joblib.load(os.path.join(self.models_dir, 'nn_model.joblib'))
            self.tfidf_matrix = joblib.load(os.path.join(self.models_dir, 'tfidf_matrix.joblib'))
            self.movies_df = joblib.load(os.path.join(self.models_dir, 'movies_metadata.joblib'))
            self.indices = joblib.load(os.path.join(self.models_dir, 'indices.joblib'))
            print("Models loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}. Please ensure 'train_model.py' has been run.")

    def get_recommendations(self, title: str, n_recommendations: int = 5) -> Union[List[Dict[str, Any]], str]:
        """Generates movie recommendations based on similarity."""
        title = str(title).strip()
        
        if title not in self.indices:
            matches = difflib.get_close_matches(title, self.indices.index, n=1, cutoff=0.6)
            if not matches:
                return f"Movie '{title}' not found and no close matches identified."
            print(f"Exact match not found. Recommending based on closest match: '{matches[0]}'")
            title = matches[0]

        idx = self.indices[title]
        
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        movie_vec = self.tfidf_matrix[idx]
        
        distances, neighbor_indices = self.nn_model.kneighbors(movie_vec, n_neighbors=n_recommendations+1)
        
        recommendations = []
        for i, neighbor_idx in enumerate(neighbor_indices[0]):
            if neighbor_idx == idx:
                continue
            
            movie_data = self.movies_df.iloc[neighbor_idx]
            recommendations.append({
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'vote_average': movie_data['vote_average'],
                'overview': movie_data['overview'],
                'similarity_score': 1 - distances[0][i]
            })
            
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

if __name__ == "__main__":
    recommender = MovieRecommender()
    
    while True:
        user_input = input("\nEnter a movie name (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
            
        recs = recommender.get_recommendations(user_input)
        
        if isinstance(recs, str):
            print(recs)
        else:
            print(f"\nRecommendations for '{user_input}':")
            for i, movie in enumerate(recs, 1):
                print(f"{i}. {movie['title']} ({movie['genres']}) - Rating: {movie['vote_average']}")
