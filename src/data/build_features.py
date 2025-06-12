import pandas as pd
import os
from sklearn.preprocessing import normalize
import json



def read_ratings(ratings_csv, data_dir="data/raw") -> pd.DataFrame:
    """
    Reads a ratings.csv from the data/raw folder.

    Parameters
    -------
    ratings_csv : str
        The csv file that will be read. Must be corresponding to a rating file.

    Returns
    -------
    pd.DataFrame
        The ratings DataFrame. Its columns are, in order:
        "userId", "movieId", "rating" and "timestamp".
    """
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))

    return data


def read_movies(movies_csv, data_dir="data/raw") -> pd.DataFrame:
    """
    Reads a movies.csv from the data/raw folder.

    Parameters
    -------
    movies_csv : str
        The csv file that will be read. Must be corresponding to a movie file.

    Returns
    -------
    pd.DataFrame
        The movies DataFrame. Its columns are binary and represent the movie genres.
    """
    # Read the CSV file
    df = pd.read_csv(os.path.join(data_dir, movies_csv))

    # Split the 'genres' column into individual genres
    genres = df["genres"].str.get_dummies(sep="|")

    # Concatenate the original movieId and title columns with the binary genre columns
    result_df = pd.concat([df[["movieId", "title"]], genres], axis=1)
    return result_df

def keep_only_liked_movies(ratings, min_rating = 4.0):
    """
    Keeps only the movies with a rating equal or above the threshold.
    """
    liked_ratings = ratings[ratings["rating"]>=min_rating]
    favorite_movies_lists = liked_ratings.groupby("userId")["movieId"].apply(list)
    return liked_ratings, favorite_movies_lists


def filter_users_with_min_likes(ratings, min_liked_movies=5):
    """
    Filters out users who liked fewer than 'min_liked_movies'.
    """
    liked_counts = ratings.groupby("userId").size()
    active_users = liked_counts[liked_counts >= min_liked_movies].index
    return ratings[ratings["userId"].isin(active_users)]


def create_user_matrix(ratings, movies, aggregation = "mean"):
    """
    Builds the user profile matrix by averaging genre vectors of liked movies per user.
    """   
    # merge the 2 tables together
    merged = ratings.merge(movies, on="movieId", how="inner")
    merged = merged.drop(columns=["movieId", "title", "rating", "timestamp"])
    user_matrix = merged.groupby("userId").agg(aggregation)
    return user_matrix

def normalize_user_matrix(user_matrix):
    """
    Normalize each user's profile vector to have unit norm (L2).
    """
    normalized_user_matrix = pd.DataFrame(
        normalize(user_matrix, norm='l2', axis=1),
        index=user_matrix.index,
        columns=user_matrix.columns
    )
    return normalized_user_matrix

def save_processed_data(movies_df, user_matrix, favorite_movies_lists, output_dir="data/processed"):
    """
    Saves the movie genre matrix and user profile matrix to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    movies_df.drop(columns=["title"]).to_csv(os.path.join(output_dir, "movie_matrix.csv"), index=False)
    user_matrix.to_csv(os.path.join(output_dir, "user_matrix.csv"))
    
    favorite_movies_lists
    with open(os.path.join(output_dir, "user_favorites.json"), "w") as f:
        json.dump(favorite_movies_lists.to_dict(), f, indent=4)
    

if __name__ == "__main__":
    # Créer le dossier "data/processed" s’il n’existe pas
    os.makedirs("data/processed", exist_ok=True)
    
    # Load raw data
    ratings = read_ratings("ratings.csv")
    movies = read_movies("movies.csv")

    # Filter and preprocess
    liked_ratings, favorite_movies_lists = keep_only_liked_movies(ratings)
    filtered_ratings = filter_users_with_min_likes(liked_ratings)
    
    # Create user profile matrix (average genre preference)
    user_matrix = normalize_user_matrix(create_user_matrix(filtered_ratings, movies))

    # Save processed data
    save_processed_data(movies, user_matrix, favorite_movies_lists)