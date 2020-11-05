import pandas as pd
import numpy as np
import csv
import os
import factorization_machine
from oneHotEncoding import get_data
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


def rmse(y, y_predicted):
    return np.sqrt(np.mean((y_predicted - y) ** 2))

def r2(y, y_predicted):
    return 1 - (np.sum(np.square(y - y_predicted))) / (np.sum(np.square(y - y.mean())))


def data_processing():
    if not os.path.isfile("netflix-prize-data/processed_data2.csv"):
        combined_data_file_name = "./netflix-prize-data/combined_data_%s.txt"

        target_f = open("netflix-prize-data/processed_data2.csv", "w+")
        for i in {1, 2, 3, 4}:
            print("Data file: " + str(i) + "/4")

            cur_movie_id = None
            with open(combined_data_file_name % i) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    if (len(row) == 1):
                        cur_movie_id = row[0][:-1]
                    else:
                        user_id = row[0]
                        rating = row[1]

                        target_f.write(user_id + "," + rating + "," + cur_movie_id + "\n")
        target_f.close()


if __name__ == '__main__':
    data_processing()
    X, ratings = get_data()

    print("One-Hot Encoding is ended")

    # do shuffling so records will be evenly distributed over the matrix
    X, ratings = shuffle(X, ratings)

    number_of_splits = 5
    factors_num = 2

    rmse_train = [0 for _ in range(number_of_splits)]
    rmse_test = [0 for _ in range(number_of_splits)]

    kf = KFold(n_splits=5)
    fm = factorization_machine.FactorizationMachine(n_iter=50)

    for i, (train_indices, test_indices) in enumerate(kf.split(X)):
        print("Fold {}".format(i + 1))

        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = ratings[train_indices], ratings[test_indices]

        fm.gradient_descent(X_train, Y_train)

        Y_pred_train = fm.predict(X_train)
        Y_pred_test = fm.predict(X_test)

        rmse_train[i] = rmse(Y_train, Y_pred_train)
        rmse_test[i] = rmse(Y_test, Y_pred_test)


    # print results
    columns = ["Names", "T1", "T2", "T3", "T4", "T5", "Mean", "Std"]
    rows = ["RMSE-train", "RMSE-test"]

    result_dataframe = pd.DataFrame(columns=columns)
    result_dataframe["Names"] = rows
    result_dataframe.set_index("Names", inplace=True)

    for i in range(number_of_splits):
        data = ([rmse_train[i], rmse_test[i]])
        result_dataframe[f"T{i + 1}"] = data

    result_dataframe["Mean"] = result_dataframe.mean(axis=1)
    result_dataframe["Std"] = result_dataframe.std(axis=1)

    print(result_dataframe.to_string())