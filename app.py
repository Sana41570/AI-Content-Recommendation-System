import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation", layout="wide")

st.title("🎬 AI-Based Movie Recommendation System")
st.write("Personalized movie suggestions using Collaborative Filtering")

# 🎬 Movie Poster Links
movie_posters = {
    "Inception":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\inception.jpg",
    "Titanic":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\titanic.jpg",
    "Avengers: Endgame":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\avengers.jpg",
    "Interstellar":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\intersellar.jpg" ,
    "Joker":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\joker.jpg",
    "The Conjuring":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\the conjuring.jpg",
    "The Dark Knight":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\the dark knight.jpg",
    "3 Idiots":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\3 idiots.jpg",
    "Dangal":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\dangal.jpg",
    "Frozen":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\frozen.jpg",
    "Bahubali":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\bahubali.jpg",
    "KGF":r"C:\Users\Sana Ansari\OneDrive\content_recommendation.py\KGF.jpg"
}

# 🎬 Larger Dataset
data = {
    'user_id': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
    'movie': [
        'Inception','Titanic','Avengers: Endgame','3 Idiots',
        'Inception','The Conjuring','Interstellar','Dangal',
        'Titanic','Interstellar','Joker','Frozen',
        'Avengers: Endgame','Joker','The Dark Knight','Bahubali',
        'KGF','3 Idiots','Dangal','Frozen'
    ],
    'rating': [5,4,5,5,4,3,5,4,5,4,4,5,5,4,5,4,5,4,5,4]
}

df = pd.DataFrame(data)

# Create user-movie matrix
user_movie_matrix = df.pivot_table(
    index='user_id',
    columns='movie',
    values='rating'
).fillna(0)

# Calculate similarity
user_similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# Recommendation function
def recommend_movies(user_id, top_n=4):
    similar_users = similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    most_similar_user = similar_users.index[0]

    similar_user_movies = user_movie_matrix.loc[most_similar_user]
    user_movies = user_movie_matrix.loc[user_id]

    recommendations = similar_user_movies[user_movies == 0]
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(top_n)

# User selection
user_id = st.selectbox("Select User ID", user_movie_matrix.index)

if st.button("Get Recommendations"):
    recs = recommend_movies(user_id)

    if len(recs) > 0:
        st.subhader("🎥 Recommended Movies For You")
        cols = st.columns(len(recs))

        for col,movie in zip(cols, recs.index):
            with col:
               
               # st.image("posters/avengers.jpg", width=150)
                st.caption(movie)
    else:
        st.write("No new recommendations available.")
    
