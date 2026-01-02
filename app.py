
import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from src.recommender import MovieRecommender

load_dotenv()

st.set_page_config(
    page_title="Cinematch",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: auto;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        background-color: #ff4b4b;
        color: white;
        border: none;
        align-items: center;
        display: flex;
        justify-content: center;
    }
    .movie-card {
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6; 
        margin-bottom: 20px;
        height: 100%;
    }
    .title-text {
        font-weight: bold;
        font-size: 1.1em;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .rating-badge {
        background-color: #e6f3ff;
        color: #0068c9;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    return MovieRecommender()

def fetch_poster(movie_id, default_path):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=2)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"{BASE_POSTER_URL}{data['poster_path']}"
    except Exception:
        pass
    
    if default_path and isinstance(default_path, str):
        return f"{BASE_POSTER_URL}{default_path}"
    
    return "https://via.placeholder.com/500x750?text=No+Image"

from src.train_model import train

def main():
    try:
        recommender = load_recommender()
    except Exception:
        recommender = None

    tab1, tab2 = st.tabs(["Recommendations", "Technical Architecture"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <h1 style='text-align: center; margin-bottom: 1rem;'>Movie Recommendation System</h1>
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <a href="https://github.com/devlucascfarias" target="_blank">
                        <img src="https://img.shields.io/badge/GitHub-devlucascfarias-181717?style=for-the-badge&logo=github" alt="GitHub">
                    </a>
                    &nbsp;&nbsp;
                    <a href="https://www.linkedin.com/in/lucas-correia-b856152b5/" target="_blank">
                        <img src="https://img.shields.io/badge/LinkedIn-Lucas%20Correia-0077B5?style=for-the-badge&logo=linkedin" alt="LinkedIn">
                    </a>
                </div>
                <p style='text-align: center; color: #666;'>
                    This system utilizes content-based filtering via TF-IDF vectorization and nearest neighbors. 
                    Type in a movie you liked, and the recommendation algorithm will suggest similar movies.
                </p>
            """, unsafe_allow_html=True)
            
            search_col, btn_col = st.columns([4, 1])
            
            with search_col:
                if recommender and recommender.indices is not None:
                    movie_list = recommender.indices.index.tolist()
                    selected_movie = st.selectbox(
                        "Select a movie:",
                        movie_list,
                        index=None,
                        placeholder="Type to search...",
                        label_visibility="collapsed"
                    )
                else:
                    selected_movie = st.text_input("Enter movie title", label_visibility="collapsed")

            with btn_col:
                if st.button("Recommend", type="primary", use_container_width=True):
                    st.session_state['selected_movie'] = selected_movie
            
        if 'selected_movie' in st.session_state and st.session_state['selected_movie']:
            selected = st.session_state['selected_movie']
            with st.spinner(f"Finding movies similar to '{selected}'..."):
                recommendations = recommender.get_recommendations(selected, n_recommendations=5)
                
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    st.markdown("---")
                    st.subheader(f"Because you watched *{selected}*")
                    
                    cols = st.columns(5)
                    for idx, movie in enumerate(recommendations):
                        with cols[idx]:
                            poster_url = fetch_poster(movie['id'], movie.get('poster_path'))
                            
                            st.image(poster_url, width='stretch')
                            st.markdown(f"<div class='title-text'>{movie['title']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<span class='rating-badge'>★ {movie['vote_average']:.1f}</span>", unsafe_allow_html=True)
                            
                            with st.expander("Overview"):
                                st.caption(movie['genres'])
                                st.write(movie['overview'][:150] + "..." if len(str(movie['overview'])) > 150 else movie['overview'])

    with tab2:
        st.header("Technical Report & Architecture")
        st.markdown("""
        This project demonstrates a production-ready **Content-Based Filtering** recommendation system.
        It leverages natural language processing and nearest neighbor algorithms to suggest movies based on metadata similarity.
        
        ### Tech Stack
        
        | Component | Technology | Role |
        |-----------|------------|------|
        | **Language** | **Python 3.10+** | Core logic and scripting. |
        | **ML Library** | **Scikit-Learn** | TF-IDF Vectorization & NearestNeighbors algorithm. |
        | **Data Processing** | **Pandas & NumPy** | Efficient data manipulation and matrix operations. |
        | **Web Framework** | **Streamlit** | Rapid frontend development with reactive components. |
        | **API Integration** | **TMDB API** | Real-time fetching of high-quality movie posters. |
        | **Serialization** | **Joblib** | Optimized model saving/loading (artifacts). |
        
        ### Machine Learning Approach
        
        1.  **Data Ingestion**: The system consumes a dataset of ~700k movies, filtered for quality (minimum 50 votes).
        2.  **Feature Engineering**: 
            - A "metadata soup" is created by combining *Title*, *Genres*, *Keywords*, *Credits*, and *Overview*.
            - Text is cleaned and processed to ensure consistency.
        3.  **Vectorization**: 
            - **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert text data into sparse numerical vectors.
            - `max_features=5000` ensures a balance between performance and accuracy.
        4.  **Similarity Search**:
            - **Unsupervised Nearest Neighbors** (Brute-force algorithm with Cosine Similarity).
            - This allows for extremely fast retrieval of similar items in a high-dimensional space.
            
        ### Architecture Highlights
        
        - **Modular Codebase**: Separated concerns into `train_model.py` (ETL & Training) and `recommender.py` (Inference logic).
        - **Inference Optimization**: 
            - Models are loaded once and cached via `st.cache_resource`.
            - Computations use sparse matrices to minimize memory footprint.
        - **Production Ready**: 
            - Error handling for missing data or API failures.
            - Type hinting and clean code practices.
            - Scalable design that can be easily containerized (Docker).
        """)

if __name__ == "__main__":
    main()
