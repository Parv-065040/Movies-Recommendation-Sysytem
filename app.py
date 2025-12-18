import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# STYLING (CLEAN & READABLE)
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
st.markdown("<h3>Foundations of Big Data Analytics using Python (FBDAP)</h3>", unsafe_allow_html=True)
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
c1, c2, c3 = st.columns(3)
c1.metric("🎥 Movies", movies.shape[0])
c2.metric("👤 Users", data["userId"].nunique())
c3.metric("⭐ Avg Rating", round(data["rating"].mean(), 2))

st.divider()

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs(
    ["🎬 Genre Recommendations", "📊 Interactive Insights", "ℹ️ About"]
)

# ======================================================
# TAB 1: GENRE-BASED RECOMMENDATION
# ======================================================
with tab1:
    st.subheader("🎭 Select a Genre")

    genre = st.selectbox(
        "Choose a genre",
        sorted(genre_cols[1:])
    )

    top_n = st.slider("Number of recommendations", 3, 10, 5)

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
        recs = recommend_by_genre(genre, top_n)
        st.success(f"Top {top_n} {genre} Movies")

        for i, row in recs.iterrows():
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
# TAB 2: INTERACTIVE INSIGHTS (PLOTLY)
# ======================================================
with tab2:
    st.subheader("📊 Ratings Distribution")

    fig1 = px.bar(
        data["rating"].value_counts().sort_index(),
        labels={"value": "Count", "index": "Rating"},
        title="Distribution of Ratings",
        color_discrete_sequence=["#f59e0b"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🎥 Top 10 Most Rated Movies")

    top_movies = (
        data.groupby("title")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        top_movies,
        x="rating",
        y="title",
        orientation="h",
        title="Top 10 Most Rated Movies",
        color="rating",
        color_continuous_scale="YlOrBr"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🎭 Genre Popularity")

    genre_counts = data[genre_cols[1:]].sum().reset_index()
    genre_counts.columns = ["Genre", "Count"]

    fig3 = px.pie(
        genre_counts,
        names="Genre",
        values="Count",
        title="Genre Distribution",
        hole=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# TAB 3: ABOUT
# ======================================================
with tab3:
    st.markdown("""
    ### 📌 Project Overview
    This **Movie Recommendation System** demonstrates:

    - Content-based filtering using genres  
    - Collaborative filtering using user ratings  
    - Cosine similarity for similarity measurement  
    - Matrix factorization (SVD) during model development  
    - Interactive deployment using Streamlit  

    ### 🚀 Key Highlights
    - Real-world Kaggle dataset  
    - Interactive Plotly visualizations  
    - Professional multi-tab dashboard  
    - Genre-based movie discovery  
    """)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>"
    "FBDA Project | MovieLens 100K | Interactive Dashboard"
    "</p>",
    unsafe_allow_html=True
)

