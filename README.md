# Movie Recommendation System (Hybrid Recommender)

An interactive and explainable **Hybrid Movie Recommendation System** developed using **Python, Streamlit, and Machine Learning**. The system combines **Content-Based Filtering** and **Collaborative Filtering** techniques to recommend movies based on genre preferences and user rating behavior.

---

## Project Overview

This project was developed as part of the **Foundations of Big Data Analytics Using Python (FBDAP)** course. It demonstrates the end-to-end design of a recommendation system, including data processing, similarity computation, hybrid scoring, and interactive visualization.

The application enables users to:

* Select a preferred movie genre
* Adjust the balance between content-based and collaborative filtering
* Generate top-N movie recommendations
* View explanations for each recommendation
* Export recommendations in CSV format

---

## Key Features

* **Hybrid Recommendation Engine**

  * Content-Based Filtering using movie genres
  * Collaborative Filtering using cosine similarity on user ratings

* **Interactive User Controls**

  * Genre selection
  * Adjustable hybrid weight (α)
  * Configurable number of recommendations

* **Explainable Recommendations**

  * Transparent logic explaining why each movie is recommended

* **External API Integration**

  * Movie poster retrieval using the TMDB API

* **Analytics Dashboard**

  * Rating distribution analysis
  * Genre popularity insights

* **Data Export**

  * CSV download for further analysis or reporting

---

## Technology Stack

* **Programming Language:** Python
* **Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Plotly
* **Similarity Metric:** Cosine Similarity
* **API:** The Movie Database (TMDB)

---

## Dataset

* **MovieLens 100K Dataset**

  * `u.data`: User–movie ratings
  * `u.item`: Movie metadata and genre information

---

## Recommendation Methodology

### Content-Based Filtering

* Uses genre information from movie metadata
* Scores movies based on their average user ratings

### Collaborative Filtering

* Constructs a user–item rating matrix
* Computes item-to-item similarity using cosine similarity

### Hybrid Scoring Formula

```
Hybrid Score = α × Content Score + (1 − α) × Collaborative Score
```

Where:

* **α** controls the trade-off between content-based and collaborative filtering

---

## Business and Practical Applications

* Personalized recommendations for OTT platforms
* Improved user engagement through data-driven content suggestions
* Demonstrates real-world recommender system implementation

---

## Academic Context

* **Course:** Foundations of Big Data Analytics Using Python (FBDAP)
* **Project Type:** Academic and Portfolio Project

---

## Author

Parv

Awantika Kholia

Ditsya Banerjee

PGDM – Big Data Analytics and Marketing

---

## Future Scope

* User-specific personalization
* Advanced recommendation techniques such as matrix factorization
* Real-time recommendation updates
* Deployment on cloud platforms

---

This project is intended for academic learning and portfolio demonstration purposes.
