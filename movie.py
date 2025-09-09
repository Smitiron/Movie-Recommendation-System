import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

print(" Loading datasets...")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

print(" Preprocessing data...")

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)
new_df = movies[['movie_id', 'title', 'tags']].copy()

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

print(" Converting movie details into vectors...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in new_df['title'].values:
        print(f"\n❌ Sorry, the movie '{movie}' is not in our database.")
        return []
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    recommendations = []
    for i in movies_list:
        recommendations.append(new_df.iloc[i[0]].title)
    return recommendations

if __name__ == "__main__":
    print("\n🎬 Welcome to the Movie Recommendation System!")
    print("✨ Type a movie name to get recommendations.")
    print("⏹ Type 'exit' anytime to quit.\n")

    while True:
        user_input = input(" Enter a movie you like: ")
        if user_input.lower() == "exit":
            print(" Exiting... Thanks for using the Movie Recommendation System! 🍿")
            break

        recs = recommend(user_input)
        if recs:
            print(f"\n Because you liked '{user_input}', you may also enjoy:")
            for idx, r in enumerate(recs, 1):
                print(f"   {idx}. {r}")
        print("\n" + "-"*50 + "\n")