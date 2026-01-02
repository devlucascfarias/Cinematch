
# Movie Recommendation System

## Overview
This repository contains a Content-Based Filtering recommendation system for movies. The application suggests movies similar to a user-provided title by analyzing metadata such as genres, overview, keywords, and credits. It utilizes Natural Language Processing (NLP) techniques and Nearest Neighbors algorithms to calculate similarity.

## Technology Stack
- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-Learn (TF-IDF Vectorizer, NearestNeighbors)
- **Data Processing:** Pandas, NumPy
- **API:** TMDB API (for fetching movie posters)
- **Model Persistence:** Joblib

## Architecture
The system operates in two main phases:

1.  **Training (Offline):**
    -   Data ingestion from `movies.csv`.
    -   Preprocessing: Cleaning text, handling missing values, and combining features into a single "soup" string.
    -   Feature Extraction: TF-IDF Vectorization (max 5000 features).
    -   Modeling: Unsupervised Nearest Neighbors with Cosine Similarity metric.
    -   Serialization: Saving model artifacts (`.joblib`) to disk.

2.  **Inference (Online):**
    -   The Streamlit app loads the pre-trained artifacts.
    -   User input is matched against the index.
    -   The model retrieves the k-nearest neighbors based on the cached TF-IDF matrix.
    -   Movie details and posters are fetched for display.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/devlucascfarias/Cinematch.git
    cd Cinematch
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Environment Setup:
    -   Create a `.env` file in the root directory.
    -   Add your TMDB API key:
        ```text
        TMDB_API_KEY=your_api_key_here
        ```

## Usage

1.  **Train the Model:**
    If running for the first time or if the dataset changes, run the training script:
    ```bash
    python src/train_model.py
    ```
    This will generate the necessary artifacts in the `models/` directory.

2.  **Run the Application:**
    Start the Streamlit interface:
    ```bash
    streamlit run app.py
    ```

## Project Structure
```text
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API Keys)
├── data/
│   └── movies.csv         # Raw dataset (not tracked in git)
├── models/                # Saved model artifacts (not tracked in git)
└── src/
    ├── train_model.py     # ETL and model training script
    └── recommender.py     # Inference logic class
```
