# 🎬 Movie Recommendation System

This project is a *Content-Based Movie Recommendation System* built
using the *TMDB 5000 Movies dataset*.\
It suggests movies similar to a given movie based on *genres, cast,
crew, keywords, and description*.

------------------------------------------------------------------------

## 🚀 Live Demo

👉 [Try the App on Streamlit
Cloud](https://smitiron-movie-recommendation-system-deploy-uso2ja.streamlit.app/)

------------------------------------------------------------------------

## 📂 Dataset

We use the [TMDB 5000 Movie
Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) which
includes: - tmdb_5000_movies.csv\
- tmdb_5000_credits.csv

Make sure both files are in the same folder as the Python script.

------------------------------------------------------------------------

## ⚙ Features

-   Loads and preprocesses movie data.\
-   Extracts important features (genre, cast, crew, keywords,
    overview).\
-   Creates a *combined tag column* for each movie.\
-   Uses *CountVectorizer* to convert text into vectors.\
-   Calculates *Cosine Similarity* between movies.\
-   Suggests *Top 5 similar movies* to the one entered by the user.\
-   Interactive CLI (Command Line Interface) with friendly outputs.

------------------------------------------------------------------------

## 🛠 Installation

Clone or download this project.\
Make sure you have Python 3.7+ installed.

Install required libraries:

``` bash
pip install pandas numpy scikit-learn
```

------------------------------------------------------------------------

## 🚀 How to Run

1.  Place the following files in the same folder:
    -   movie_recommender.py (or movie_recommender_interactive.py)\
    -   tmdb_5000_movies.csv\
    -   tmdb_5000_credits.csv
2.  Run the program:\

``` bash
python movie_recommender.py
```

------------------------------------------------------------------------

## 💡 Example Usage

When you run the program, you'll see:

    🎬 Welcome to the Movie Recommendation System!
    ✨ Type a movie name to get recommendations.
    ⏹ Type 'exit' anytime to quit.

    👉 Enter a movie you like: Avatar
    🎥 Because you liked 'Avatar', you may also enjoy:
       1. Guardians of the Galaxy
       2. John Carter
       3. Star Trek
       4. Star Wars: The Clone Wars
       5. Star Trek Into Darkness

👉 Enter another movie:

    👉 Enter a movie you like: The Dark Knight Rises
    🎥 Because you liked 'The Dark Knight Rises', you may also enjoy:
       1. The Dark Knight
       2. Batman Begins
       3. Man of Steel
       4. Superman Returns
       5. The Prestige

Type `exit` to quit.

------------------------------------------------------------------------

## 📖 Concepts Used

-   Data Preprocessing (handling text and categorical data)\
-   Natural Language Processing (CountVectorizer)\
-   Machine Learning (Cosine Similarity for recommendations)\
-   Interactive CLI Application
