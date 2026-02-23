import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 AI Content Recommendation System")

# Sample dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],
    'movie': [
        'Action Movie', 'Comedy Show', 'Romantic Movie',
        'Action Movie', 'Horror Show',
        'Comedy Show', 'Romantic Movie',
        'Action Movie', 'Comedy Show'
    ],
    'rating': [5, 4, 5, 4, 3, 5, 4, 4, 5]
}

df = pd.DataFrame(data)

# Create pivot table
user_movie_matrix = df.pivot_table(index='user_id',
                                    columns='movie',
                                    values='rating').fillna(0)

# Calculate similarity
user_similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(user_similarity,
                             index=user_movie_matrix.index,
                             columns=user_movie_matrix.index)

# Recommendation function
def recommend_movies(user_id):
    similar_users = similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    most_similar_user = similar_users.index[0]

    similar_user_movies = user_movie_matrix.loc[most_similar_user]
    user_movies = user_movie_matrix.loc[user_id]

    recommendations = similar_user_movies[user_movies == 0]
    return recommendations.sort_values(ascending=False)

# User input
user_id = st.selectbox("Select User ID", user_movie_matrix.index)

if st.button("Get Recommendations"):
    recs = recommend_movies(user_id)
    st.write("### Recommended Content:")
    st.write(recs)
    

