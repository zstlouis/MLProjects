import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load csv into a dataframe
df = pd.read_csv('movies.csv')

# check if null values exist
print(df.isnull().sum())

# extract relevant features from the dataset
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print(df[selected_features])

# replace null values with null string
for feature in selected_features:
    df[feature] = df[feature].fillna('')

# view that null values have been removed from the dataframe
# now appear as an empty string
print(df.isnull().sum())

# combine the selected features
combined_features = df["genres"] + ' ' + df["keywords"] + ' ' + df["tagline"] + ' ' + df['cast'] + ' ' + df['director']
print(combined_features)

# convert to feature vectors (Term Frequency Inverse Document Frequency)
# term freq -> number of times word appears in a doc
# inverse document freq -> number of docs word appears in
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

print(feature_vectors)

# Cosine Similarity
# get similarity score
# will compare each movie with all other movies
similarity = cosine_similarity(feature_vectors)
print(similarity)
print(similarity.shape)

# get a movie from the user
movie_input = input('Enter in a movie: ')

# creating a list with all the movies names from dataset
movie_names_list = np.asarray(df['title'])
# print(movie_names_list)

# finding matching movies
# .get_close_matches(word, possibilities)
find_match = difflib.get_close_matches(movie_input, movie_names_list)
print(find_match)

# get the first movie returned
close_match = find_match[0]

# find index of the movie to then use to find similarity score of other movies
index_of_movie = df[df.title == close_match]['index'].values[0]
print(index_of_movie)

# get the similarity score
similarity_score = list(enumerate(similarity[index_of_movie]))
print(similarity_score)

# return top 5 similarity scores
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
print(sorted_similar_movies[1:6])

# print the names of similar movies
i = 1
print('Movies suggested for you: \n')
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = df[df.index == index]['title'].values[0]
    if (i < 30):
        print(i, title_from_index)
        i += 1
