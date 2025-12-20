import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import re
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
FALLBACK_POSTER = "https://via.placeholder.com/300x450.png?text=No+Poster"

GROUP_ID = 244060
SAMPLE_SIZE = 10001

# ======================================================
# STYLING
# ======================================================
st.markdown("""
<style>
body { background-color: #f4f6f9; }
h1 { text-align:center; color:#111827; }
h3 { text-align:center; color:#374151; }
.movie-card {
    background-color:#ffffff;
    padding:16px;
    border-radius:12px;
    margin-bottom:16px;
    box-shadow:0px 6px 14px rgba(0,0,0,0.08);
}
.movie-title {
    font-size:20px;
    font-weight:bold;
}
.movie-rating {
    color:#047857;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Hybrid Recommendation Dashboard (FBDA)</h3>", unsafe_allow_html=True)
st.divider()

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "u.data", sep="\t",
        names=["userId", "movieId", "rating", "timestamp"]
    ).sample(n=SAMPLE_SIZE, random_state=GROUP_ID)

    movies = pd.read_csv(
        "u.item", sep="|",
        encoding="latin-1", header=None
    )

    genre_cols = [
        "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
        "Docume
