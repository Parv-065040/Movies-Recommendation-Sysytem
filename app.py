import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ======================================================
# PAGE CONFIG & STYLING
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0f1117;
}
.main {
    background-color: #0f1117;
}
h1 {
    color: #F5C518;
    text-align: center;
}
h3 {
    color: #FFFFFF;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    background-color: #F5C518;
    color: black;
    font-weight: bold;
    border-radius: 10px;
}
.stSelectbox label {
    color: #FFFFFF;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align:center;'>Foundations of Big Data Analytics with Python (FBDA)</h3>",
    unsafe_allow_html=True
)

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

    movies = movies[[0, 1]]
    movies.columns = ["movieId", "title"]

    data = ratings.merge(movies, on="movieId")
    return data, movies

with st.spinner("🔄 Loading data..."):
    data, movies = load_data()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("📌 Project Details")
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
# USER-ITEM MATRIX
# ======================================================
user_item = data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

sparse_matrix = csr_matrix(user_item.values)

# ======================================================
# VALID MOVIES (CRITICAL FIX)
# ======================================================
valid_movie_ids = set(user_item.columns)
valid_movies = movies[movies["movieId"].isin(valid_movie_ids)]

# ======================================================
# ITEM-ITEM COSINE SIMILARITY
# ======================================================
with st.spinner("⚙️ Computing similarity matrix..."):
    item_similarity = cosine_similarity(sparse_matrix.T)

# ======================================================
# MAIN UI
# ======================================================
st.subheader("🎥 Choose a Movie You Like")

col1, col2 = st.columns([2, 1])

with col1:
    movie_selected = st.selectbox(
        "Select a movie:",
        valid_movies["title"].sort_values().values
    )

with col2:
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=3,
        max_value=10,
        value=5
    )

# ======================================================
# RECOMMENDATION FUNCTION
# ======================================================
def recommend_movies(movie_title, top_n):
    movie_id = valid_movies[
        valid_movies["title"] == movie_title
    ]["movieId"].values[0]

    movie_idx = user_item.columns.get_loc(movie_id)

    similarity_scores = list(enumerate(item_similarity[movie_idx]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

    recommended_ids = [user_item.columns[i[0]] for i in similarity_scores]
    return movies[movies["movieId"].isin(recommended_ids)]

# ======================================================
# BUTTON ACTION
# ======================================================
if st.button("✨ Recommend Movies"):
    with st.spinner("🎯 Finding the best recommendations for you..."):
        recommendations = recommend_movies(
            movie_selected,
            num_recommendations
        )

    st.success("✅ Recommendations Ready!")

    st.subheader("🍿 Recommended Movies For You")

    for _, row in recommendations.iterrows():
        st.markdown(
            f"""
            <div style="
                background-color:#1c1e26;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
                border-left:6px solid #F5C518;
            ">
            🎬 <b>{row['title']}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "Built using Streamlit | MovieLens 100K | FBDA Project"
    "</p>",
    unsafe_allow_html=True
)

