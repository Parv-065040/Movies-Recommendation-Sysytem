import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 14px;
    border-left: 6px solid #f59e0b;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.08);
}
.movie-title {
    font-size: 18px;
    font-weight: bold;
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
st.markdown("<h3>Hybrid Recommendation Engine (FBDA)</h3>", unsafe_allow_html=True)
st.divider()

# ======================================================
# CONSTANTS
# ======================================================
GROUP_ID = 244060
SAMPLE_SIZE = 10001

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
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Hybrid Settings")
    selected_genre = st.selectbox("Select Genre", sorted(genre_cols[1:]))
    alpha = st.slider(
        "Weight: Content vs Collaborative",
        0.0, 1.0, 0.6,
        help="Higher value favors genre preference"
    )
    top_n = st.slider("Number of Recommendations", 3, 10, 5)

# ======================================================
# HYBRID RECOMMENDATION LOGIC
# ======================================================
def hybrid_recommendation(genre, alpha, top_n):
    # Content-based score
    genre_movies = movies[movies[genre] == 1]
    avg_ratings = (
        data.groupby("movieId")["rating"]
        .mean()
        .reset_index(name="avg_rating")
    )

    content_scores = genre_movies.merge(
        avg_ratings, on="movieId", how="left"
    )

    content_scores["content_score"] = content_scores["avg_rating"] / 5

    # Collaborative score
    movie_indices = list(user_item.columns)
    sim_scores = np.mean(item_similarity, axis=0)

    collab_scores = pd.DataFrame({
        "movieId": movie_indices,
        "collab_score": sim_scores
    })

    # Hybrid merge
    hybrid = content_scores.merge(collab_scores, on="movieId")
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
    st.success("Hybrid Recommendations Ready")

    for i, row in results.iterrows():
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
    "Hybrid Recommendation System | FBDA | MovieLens 100K"
    "</p>",
    unsafe_allow_html=True
)
