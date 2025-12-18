import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# CLEAN & READABLE STYLING (TEXT VISIBILITY FIXED)
# ======================================================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main {
    background-color: #f4f6f9;
}
h1 {
    color: #1f2937;
    text-align: center;
}
h3 {
    color: #374151;
    text-align: center;
}
.movie-card {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
    border-left: 6px solid #f59e0b;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
.movie-title {
    font-size: 18px;
    font-weight: bold;
    color: #111827;
}
.movie-rating {
    color: #065f46;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Foundations of Big Data Analytics with Python (FBDAP)</h3>", unsafe_allow_html=True)
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
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("📌 Project Info")
    st.markdown("""
    **Course:** FBDA  
    **Dataset:** MovieLens 100K  
    **Sample Size:** 10,001  
    **Group ID:** 244060  
    """)
    st.markdown("---")
    st.markdown("👥 **Group Members**")
    st.markdown("""
    - 065024  
    - 065040  
    - 065060  
    """)

# ======================================================
# GENRE SELECTION
# ======================================================
st.subheader("🎭 Select a Genre")

selected_genre = st.selectbox(
    "Choose a genre you like:",
    sorted(genre_cols[1:])  # remove 'unknown'
)

# ======================================================
# GENRE-BASED RECOMMENDATION LOGIC
# ======================================================
def recommend_by_genre(genre, top_n=5):
    genre_movies = data[data[genre] == 1]

    avg_ratings = (
        genre_movies.groupby("title")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "Average Rating"})
        .sort_values(by="Average Rating", ascending=False)
    )

    return avg_ratings.head(top_n)

# ======================================================
# BUTTON ACTION
# ======================================================
if st.button("🎯 Recommend Movies"):
    recommendations = recommend_by_genre(selected_genre)

    st.success(f"Top {len(recommendations)} {selected_genre} Movies")

    for _, row in recommendations.iterrows():
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="movie-title">🎬 {row['title']}</div>
                <div class="movie-rating">⭐ Rating: {row['Average Rating']:.2f}</div>
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
    "Genre-based Recommendation System | MovieLens 100K | FBDA Project"
    "</p>",
    unsafe_allow_html=True
)



