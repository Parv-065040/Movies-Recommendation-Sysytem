import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "d7db7cc5e131166fd7dd007d0ded47d2"
FALLBACK_POSTER = "https://via.placeholder.com/300x450.png?text=No+Poster"

GROUP_ID = 244060
SAMPLE_SIZE = 10001

# ======================================================
# THEME-SAFE STYLING (LIGHT + DARK)
# ======================================================
st.markdown("""
<style>
/* Use Streamlit theme variables */
:root {
    --card-bg: var(--secondary-background-color);
    --card-border: rgba(120,120,120,0.25);
}

/* Headings */
h1, h3 {
    text-align: center;
}

/* Result header */
.result-header {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Movie card */
.movie-card {
    background-color: var(--card-bg);
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid var(--card-border);
}

/* Movie title */
.movie-title {
    font-size: 20px;
    font-weight: bold;
}

/* Rating text */
.movie-rating {
    font-weight: bold;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Hybrid Recommendation Dashboard (FBDAP)</h3>", unsafe_allow_html=True)
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
# USER-ITEM MATRIX
# ======================================================
user_item = data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

item_similarity = cosine_similarity(csr_matrix(user_item.values).T)

# ======================================================
# POSTER FETCH
# ======================================================
def clean_title(title):
    return re.sub(r"\(\d{4}\)", "", title).strip()

@st.cache_data
def get_movie_poster(title):
    try:
        query = clean_title(title)
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": query}
        r = requests.get(url, params=params, timeout=5).json()

        if r.get("results"):
            poster = r["results"][0].get("poster_path")
            if poster:
                return f"https://image.tmdb.org/t/p/w500{poster}"
    except:
        pass
    return FALLBACK_POSTER

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Controls")
    genre = st.selectbox("Select Genre", sorted(genre_cols[1:]))
    alpha = st.slider("Content vs Collaborative Weight", 0.0, 1.0, 0.6)
    top_n = st.slider("Number of Recommendations", 3, 10, 5)

# ======================================================
# HYBRID RECOMMENDER
# ======================================================
def hybrid_recommendation(genre, alpha, top_n):
    genre_movies = movies[movies[genre] == 1]

    avg_ratings = (
        data.groupby("movieId")["rating"]
        .mean()
        .reset_index(name="avg_rating")
    )

    content = genre_movies.merge(avg_ratings, on="movieId")
    content["content_score"] = content["avg_rating"] / 5

    collab_scores = pd.DataFrame({
        "movieId": user_item.columns,
        "collab_score": np.mean(item_similarity, axis=0)
    })

    hybrid = content.merge(collab_scores, on="movieId")
    hybrid["hybrid_score"] = (
        alpha * hybrid["content_score"] +
        (1 - alpha) * hybrid["collab_score"]
    )

    return hybrid.sort_values("hybrid_score", ascending=False).head(top_n)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs(
    ["🎯 Hybrid Recommendations", "📊 Data Insights", "ℹ️ About"]
)

# ---------------- TAB 1 ----------------
with tab1:
    if st.button("🚀 Generate Recommendations"):
        results = hybrid_recommendation(genre, alpha, top_n)

        st.markdown(
            f"<div class='result-header'>🎬 Top {top_n} {genre} Movies Recommended for You</div>",
            unsafe_allow_html=True
        )

        for i, row in results.iterrows():
            poster = get_movie_poster(row["title"])

            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(poster, width=150)
            with col2:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">#{i+1} {row['title']}</div>
                        <div class="movie-rating">
                            ⭐ Rating: {row['avg_rating']:.2f}<br>
                            🔀 Hybrid Score: {row['hybrid_score']:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("📊 Interactive Analytics")

    fig1 = px.bar(
        data["rating"].value_counts().sort_index(),
        labels={"value":"Count","index":"Rating"},
        title="Ratings Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    genre_counts = data[genre_cols[1:]].sum().reset_index()
    genre_counts.columns = ["Genre","Count"]

    fig2 = px.pie(
        genre_counts,
        names="Genre",
        values="Count",
        hole=0.4,
        title="Genre Popularity"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown("""
    ### 📌 About the Project
    This dashboard implements a **Hybrid Recommendation System** using:
    - Content-Based Filtering (Genres)
    - Collaborative Filtering (User Ratings)
    - Cosine Similarity
    - TMDB API for posters
    - Interactive Plotly visualizations

    Developed for **Foundations of Big Data Analytics with Python (FBDA)**.
    """)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.markdown(
    "<p style='text-align:center; opacity:0.7;'>"
    "Hybrid Recommendation Dashboard | FBDA Project"
    "</p>",
    unsafe_allow_html=True
)
