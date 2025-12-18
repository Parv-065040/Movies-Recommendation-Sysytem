import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System (FBDA Project)")

GROUP_ID = 244060
SAMPLE_SIZE = 10001

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "u.data",
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"]
    ).sample(n=SAMPLE_SIZE, random_state=GROUP_ID)

    movies = pd.read_csv(
        "u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )
    movies = movies[[0, 1]]
    movies.columns = ["movieId", "title"]

    data = ratings.merge(movies, on="movieId")
    return data, movies

data, movies = load_data()

# ---------------------------
# USER-ITEM MATRIX
# ---------------------------
user_item = data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

sparse_matrix = csr_matrix(user_item.values)

# ---------------------------
# ITEM-ITEM COSINE SIMILARITY
# ---------------------------
@st.cache_resource
def compute_similarity(matrix):
    return cosine_similarity(matrix.T)

item_similarity = compute_similarity(sparse_matrix)

# ---------------------------
# UI
# ---------------------------
movie_list = movies["title"].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

# ---------------------------
# RECOMMENDATION FUNCTION
# ---------------------------
def recommend_movies(movie_title, top_n=5):
    movie_id = movies[movies["title"] == movie_title]["movieId"].values[0]
    movie_idx = list(user_item.columns).index(movie_id)

    similarity_scores = list(enumerate(item_similarity[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommended_ids = [user_item.columns[i[0]] for i in similarity_scores]
    return movies[movies["movieId"].isin(recommended_ids)]

# ---------------------------
# BUTTON
# ---------------------------
if st.button("Recommend Movies"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("🎯 Recommended Movies")
    st.dataframe(recommendations.reset_index(drop=True))

st.caption("FBDA Project | Group ID: 244060 | MovieLens 100K Dataset")
