import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, vstack


def get_data():
    data_file_path = "netflix-prize-data/processed_data2.csv"

    df = pd.read_csv(data_file_path, header=None, names=['User_Id', 'Rating', 'Movie_Id'])
    # print(df.iloc[::5000000, :])

    encoder = OneHotEncoder(categories='auto')

    # (number_of_ratings x number_of_users)
    one_hot_user_matrix = encoder.fit_transform(np.asarray(df['User_Id']).reshape(-1, 1))
    print("One-hot user matrix shape: " + str(one_hot_user_matrix.shape))

    # (number_of_ratings x number_of_movie_ids)
    one_hot_movie_matrix = encoder.fit_transform(np.asarray(df['Movie_Id']).reshape(-1, 1))
    print("One-hot movie matrix shape: " + str(one_hot_movie_matrix.shape))

    # data to predict
    ratings = np.asarray(df['Rating']).reshape(-1, 1)

    # ones for w0
    lines_number = ratings.shape[0]
    ones = np.ones(shape=(lines_number, 1))

    # train data in CSR format
    X = hstack([ones, one_hot_user_matrix, one_hot_movie_matrix]).tocsr()

    return X, ratings
