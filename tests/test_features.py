import pytest
import pandas as pd
from src.features.build_features import (
    read_ratings,
    read_movies,
    keep_only_liked_movies,
    filter_users_with_min_likes,
    create_user_matrix,
    normalize_user_matrix,
    save_processed_data,
)

def test_read_ratings(tmp_path):
    test_csv = tmp_path / "ratings.csv"
    test_csv.write_text("userId,movieId,rating,timestamp\n1,101,4.5,964982703\n2,102,3.0,964982703")
    ret = read_ratings(ratings_csv=test_csv.name, data_dir=tmp_path)
    assert isinstance(ret, pd.DataFrame)
    assert list(ret.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert len(ret) == 2

def test_read_movies(tmp_path):
    test_csv = tmp_path / "movies.csv"
    test_csv.write_text("movieId,title,genres\n101,Movie1,Action|Comedy\n102,Movie2,Drama")
    ret = read_movies(movies_csv=test_csv.name, data_dir=tmp_path)
    assert isinstance(ret, pd.DataFrame)
    assert "Action" in ret.columns
    assert "Drama" in ret.columns
    assert len(ret) == 2

def test_keep_only_liked_movies():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [101, 102, 103],
        "rating": [4.5, 3.0, 5.0],
    })
    liked_ratings, favorite_movies_lists = keep_only_liked_movies(ratings, min_rating=4.0)
    assert len(liked_ratings) == 2
    assert favorite_movies_lists[1] == [101]
    assert favorite_movies_lists[2] == [103]

def test_filter_users_with_min_likes():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 2, 3],
        "movieId": [101, 102, 103, 104, 105, 106],
        "rating": [4.5, 3.0, 5.0, 4.0, 4.0, 5.0],
    })
    filtered_ratings = filter_users_with_min_likes(ratings, min_liked_movies=2)
    assert len(filtered_ratings["userId"].unique()) == 2
    assert 3 not in filtered_ratings["userId"].values

def test_create_user_matrix():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [101, 102, 103],
        "rating": [4.5, 3.0, 5.0],
        "timestamp": [964982703, 964982704, 964982705],
    })
    movies = pd.DataFrame({
        "movieId": [101, 102, 103],
        "title": ["Movie1", "Movie2", "Movie3"],
        "Action": [1, 0, 0],
        "Comedy": [0, 1, 0],
        "Drama": [0, 0, 1],
    })
    user_matrix = create_user_matrix(ratings, movies, aggregation="mean")
    assert user_matrix.shape == (2, 3)
    assert "Action" in user_matrix.columns

def test_normalize_user_matrix():
    user_matrix = pd.DataFrame({
        "Action": [3, 0],
        "Comedy": [4, 0],
        "Drama": [0, 5],
    }, index=[1, 2])
    normalized = normalize_user_matrix(user_matrix)
    assert pytest.approx(normalized.loc[1, "Action"]) == 0.6
    assert pytest.approx(normalized.loc[1, "Comedy"]) == 0.8
    assert pytest.approx(normalized.loc[2, "Drama"]) == 1.0