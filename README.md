# ybi-project: Movie Recommendation System Project
Step 1: Import Libraries

python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


#### Step 2: Import Data

Assume you have a dataset movies.csv containing movie titles, genres, and other relevant information.

python
movies = pd.read_csv('movies.csv')


#### Step 3: Data Preprocessing

Clean and preprocess the data if necessary. For simplicity, let's assume your data is clean and structured.

#### Step 4: Vectorize Text Data

python
# Combine relevant text columns into a single string
movies['combined_features'] = movies['genre'] + ' ' + movies['director'] + ' ' + movies['actors']

# Initialize CountVectorizer to convert text data into matrix of token counts
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined_features'])


#### Step 5: Compute Similarity Matrix

python
# Compute cosine similarity matrix from the count matrix
cosine_sim = cosine_similarity(count_matrix)


#### Step 6: Define Function to Recommend Movies

python
def recommend_movies(movie_title, cosine_sim):
    # Find index of the movie that matches the title
    idx = movies[movies['title'] == movie_title].index[0]

    # Get pairwise similarity scores with all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 10 similar movies (excluding itself)
    sim_scores = sim_scores[1:11]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return top 10 recommended movies
    return movies['title'].iloc[movie_indices]


#### Step 7: Test the Recommendation System

python
# Example usage
movie_title = 'Inception'
recommended_movies = recommend_movies(movie_title, cosine_sim)
print(f"Recommended movies for '{movie_title}':")
print(recommended_movies)
