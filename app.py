import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "d7db7cc5e131166fd7dd007d0ded47d2"

GROUP_ID = 244060
SAMPLE_SIZE = 10001

# ======================================================
# STYLING
# ======================================================
st.markdown("""
<style>
body { background-color: #f4f6f9; }
h1 { color: #111827; text-align: center; }
h3 { color: #374151; text-align: center; }
.movie-card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 16px;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.08);
}
.movie-title {
    font-size: 20px;
    font-weight: bold;
    color: #111827;
}
.movie-rating {
    color: #047857;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Hybrid Recommendation Engine with Posters (FBDA)</h3>", unsafe_allow_html=True)
st.divider()

# ======================================================
# LOAD DATA
# ======================================================
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

    genre_cols = [
        "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
        "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]

    movies = movies[[0,1] + list(range(5,24))]
    movies.columns = ["movieId","title"] + genre_cols

    data = ratings.merge(movies, on="movieId")
    return data, movies, genre_cols

data, movies, genre_cols = load_data()

# ======================================================
# KPI METRICS
# ======================================================
c1, c2, c3 = st.columns(3)
c1.metric("🎥 Movies", movies.shape[0])
c2.metric("👤 Users", data["userId"].nunique())
c3.metric("⭐ Avg Rating", round(data["rating"].mean(), 2))

st.divider()

# ======================================================
# USER-ITEM MATRIX (COLLABORATIVE FILTERING)
# ======================================================
user_item = data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

sparse_matrix = csr_matrix(user_item.values)
item_similarity = cosine_similarity(sparse_matrix.T)

# ======================================================
# POSTER FETCH FUNCTION (TMDB)
# ======================================================
@st.cache_data
def get_movie_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(url, params=params, timeout=5).json()

        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
with st.sidebar:
    st.header("⚙️ Hybrid Controls")
    selected_genre = st.selectbox("Select Genre", sorted(genre_cols[1:]))
    alpha = st.slider(
        "Content vs Collaborative Weight",
        0.0, 1.0, 0.6,
        help="Higher value gives more importance to genre similarity"
    )
    top_n = st.slider("Number of Recommendations", 3, 10, 5)

# ======================================================
# HYBRID RECOMMENDER LOGIC
# ======================================================
def hybrid_recommendation(genre, alpha, top_n):
    # Content-based score
    genre_movies = movies[movies[genre] == 1]

    avg_ratings = (
        data.groupby("movieId")["rating"]
        .mean()
        .reset_index(name="avg_rating")
    )

    content = genre_movies.merge(avg_ratings, on="movieId", how="left")
    content["content_score"] = content["avg_rating"] / 5

    # Collaborative score
    collab_scores = pd.DataFrame({
        "movieId": user_item.columns,
        "collab_score": np.mean(item_similarity, axis=0)
    })

    # Hybrid score
    hybrid = content.merge(collab_scores, on="movieId")
    hybrid["hybrid_score"] = (
        alpha * hybrid["content_score"] +
        (1 - alpha) * hybrid["collab_score"]
    )

    return hybrid.sort_values(
        by="hybrid_score", ascending=False
    ).head(top_n)

# ======================================================
# MAIN OUTPUT
# ======================================================
st.subheader("🎯 Hybrid Movie Recommendations")

if st.button("🚀 Generate Recommendations"):
    results = hybrid_recommendation(selected_genre, alpha, top_n)
    st.success("Recommendations Ready")

    for i, row in results.iterrows():
        poster = get_movie_poster(row["title"])

        col1, col2 = st.columns([1, 3])
        with col1:
            if poster:
                st.image(poster, width=140)
            else:
                st.write("🎞️ No Poster")

        with col2:
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">#{i+1} 🎬 {row['title']}</div>
                    <div class="movie-rating">
                        ⭐ Avg Rating: {row['avg_rating']:.2f}<br>
                        🔀 Hybrid Score: {row['hybrid_score']:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>"
    "Hybrid Movie Recommendation System with Posters | FBDA Project"
    "</p>",
    unsafe_allow_html=True
)
