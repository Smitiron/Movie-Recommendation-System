import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ---------------------
# Data Loading & Preprocessing
# ---------------------
@st.cache_resource
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert3(obj):
        return [i['name'] for i in ast.literal_eval(obj)[:3]]

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = (
        movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    )

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

new_df, similarity = load_data()

# ---------------------
# Recommendation Function
# ---------------------
def recommend(movie):
    if movie not in new_df['title'].values:
        return []
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]
    recommendations = [(new_df.iloc[i[0]].title, i[1]) for i in movies_list]
    return recommendations

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
    .title {
        font-size: 40px; 
        font-weight: bold; 
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .movie-card {
        background: linear-gradient(135deg, #fdfbfb, #ebedee);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 6px 16px rgba(0,0,0,0.2);
    }
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    .movie-score {
        font-size: 14px;
        color: #888;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>🍿 Movie Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find movies similar to your favorite one!</div>", unsafe_allow_html=True)

movie_list = new_df['title'].values
selected_movie = st.selectbox("🎥 Select a movie:", movie_list)

if st.button("Recommend"):
    recs = recommend(selected_movie)
    if recs:
        st.subheader(f"✨ Because you liked *{selected_movie}*, you may also enjoy:")

        cols = st.columns(5)  # show 5 recommendations side by side
        for idx, (title, score) in enumerate(recs):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">{title}</div>
                        <div class="movie-score">🔹 Similarity: {score:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.error("❌ Sorry, movie not found in the database.")