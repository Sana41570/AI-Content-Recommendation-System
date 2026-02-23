import pandas as pd
import numpy as np
# Sample user-movie rating data
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
print(df)
# Create pivot table (User-Movie Matrix)
user_movie_matrix = df.pivot_table(index='user_id',
                                    columns='movie',
                                    values='rating').fillna(0)

print(user_movie_matrix)
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(user_similarity,
                             index=user_movie_matrix.index,
                             columns=user_movie_matrix.index)

print(similarity_df)
def recommend_movies(user_id, top_n=2):
    
    # Find similar users
    similar_users = similarity_df[user_id].sort_values(ascending=False)
    
    # Remove the same user
    similar_users = similar_users.drop(user_id)
    
    # Get the most similar user
    most_similar_user = similar_users.index[0]
    
    print(f"Most similar user to User {user_id} is User {most_similar_user}")
    
    # Movies watched by similar user
    similar_user_movies = user_movie_matrix.loc[most_similar_user]
    
    # Movies already watched by current user
    user_movies = user_movie_matrix.loc[user_id]
    
    # Recommend movies not watched yet
    recommendations = similar_user_movies[user_movies == 0].sort_values(ascending=False)
    
    return recommendations.head(top_n)
# Test recommendation for User 2
recommended = recommend_movies(user_id=2)
print("Recommended Movies:")
print(recommended)