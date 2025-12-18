import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# STYLING (CLEAN, PROFESSIONAL, READABLE)
# ======================================================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
h1 {
    color: #111827;
    text-align: center;
}
h2, h3 {
    color: #1f2937;
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
st.markdown("<h3 style='text-align:center;'>Foundations of Big Data Analytics using Python (FBDAP)</h3>", unsafe_allow_html=True)
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
# KPI METRICS
# ======================================================
col1, col2, col3 = st.columns(3)
col1.metric("🎥 Total Movies", movies.shape[0])
col2.metric("👤 Users", data["userId"].nunique())
col3.metric("⭐ Avg Rating", round(data["rating"].mean(), 2))

st.divider()

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs(
    ["🎬 Genre-Based Recommendation", "📊 Data Insights", "ℹ️ About Project"]
)

# ======================================================
# TAB 1: GENRE-BASED RECOMMENDER
# ======================================================
with tab1:
    st.subheader("🎭 Select a Genre")

    selected_genre = st.selectbox(
        "Choose a genre:",
        sorted(genre_cols[1:]),
        help="Select a genre to get top-rated movies"
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    def recommend_by_genre(genre, top_n):
        genre_movies = data[data[genre] == 1]

        avg_ratings = (
            genre_movies.groupby("title")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "Average Rating"})
            .sort_values(by="Average Rating", ascending=False)
        )

        return avg_ratings.head(top_n)

    if st.button("🎯 Recommend Movies"):
        recommendations = recommend_by_genre(selected_genre, top_n)

        st.success(f"Top {top_n} {selected_genre} Movies")

        for i, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">#{i+1} 🎬 {row['title']}</div>
                    <div class="movie-rating">⭐ Rating: {row['Average Rating']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================================================
# TAB 2: DATA INSIGHTS
# ======================================================
with tab2:
    st.subheader("📊 Ratings Distribution")

    fig, ax = plt.subplots()
    data["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("🎥 Top 10 Most Rated Movies")

    top_movies = (
        data.groupby("title")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )

    fig2, ax2 = plt.subplots()
    top_movies.plot(kind="barh", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of Ratings")
    st.pyplot(fig2)

# ======================================================
# TAB 3: ABOUT PROJECT
# ======================================================
with tab3:
    st.subheader("ℹ️ Project Overview")

    st.markdown("""
    This **Movie Recommendation System** was developed as part of the  
    **Foundations of Big Data Analytics with Python (FBDA)** course.

    ### Techniques Used
    - **Content-Based Filtering** using movie genres  
    - **Collaborative Filtering** using user ratings  
    - **Cosine Similarity** for measuring movie similarity  
    - **Matrix Factorization (SVD)** during model development  
    - **Streamlit** for interactive deployment  

    ### Key Features
    - Real-world dataset from Kaggle
    - Interactive dashboard
    - Clean and professional UI
    - Fully deployable web application
    """)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>"
    "FBDA Project | MovieLens 100K | Deployed with Streamlit"
    "</p>",
    unsafe_allow_html=True
)
