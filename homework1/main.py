from sklearn import preprocessing
from sklearn.model_selection import KFold
import numpy as np

def rmse(y, y_predicted):
    return np.sqrt(np.mean((y_predicted - y) ** 2))

def r2(y, y_predicted):
    return 1 - (np.sum(np.square(y - y_predicted))) / (np.sum(np.square(y - y.mean())))

def gradient_descent(x, y):
    w = np.zeros((x.shape[1], 1))
    delta_w = np.inf

    i = 0
    etha = 0.01

    while delta_w > 1e-4 and i < 35000:
        error = x.dot(w) - y
        gradient = x.T.dot(error) / len(y)
        w_next = w - etha * gradient
        i += 1
        delta_w = np.linalg.norm(w - w_next)
        w = w_next

    return w

def gradient(x, y):
    N = len(x)
    w = np.zeros((x.shape[1], 1))
    eta = 0.01

    maxIteration = 300
    for i in range(maxIteration):
        error = x.dot(w) - y
        gradient = x.T.dot(error) / N
        w = w - eta * gradient
    return w


if __name__ == '__main__':
    file_name = "Features_Variant_1.csv"
    raw_data = np.loadtxt(file_name, delimiter=",")
    Y = raw_data[:, 53]
    Y = Y.reshape(Y.size, 1)
    X = raw_data[:, 0:53]

    weights = [[0 for _ in range(55)] for _ in range(5)]
    rmse_train = [0 for _ in range(5)]
    rmse_test = [0 for _ in range(5)]
    r2s_train = [0 for _ in range(5)]
    r2s_test = [0 for _ in range(5)]

    scaler = preprocessing.MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    X_norm = np.hstack((np.ones_like(Y), X_norm))

    kf = KFold(n_splits=5)

    for i, (train_indices, test_indices) in enumerate(kf.split(X_norm)):
        out = "Fold {}".format(i)
        print(out)

        X_train, X_test = X_norm[train_indices], X_norm[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]

        weights[i] = gradient_descent(X_train, Y_train)
        W = gradient(X_train, Y_train)
        Y_pred_test = X_test.dot(weights[i])
        Y_pred_train = X_train.dot(weights[i])

        rmse_train[i] = rmse(Y_train, Y_pred_train)
        rmse_test[i] = rmse(Y_test, Y_pred_test)

        r2s_train[i] = r2(Y_train, Y_pred_train)
        r2s_test[i] = r2(Y_test, Y_pred_test)

        weights[i] = scaler.fit_transform(weights[i])

    header_format = "{:>10} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} \n"
    row_format = "{:>10} {:>20f} {:>20f} {:>20f} {:>20f} {:>20f} {:>20f} {:>20f} \n"
    columns_names = ["--", "T1", "T2", "T3", "T4", "T5", "E", "STD"]

    f = open("results.txt", "w")
    f.write(header_format.format(*columns_names))
    f.write(row_format.format("rmse-train", *rmse_train, np.mean(rmse_train), np.std(rmse_train)))
    f.write(row_format.format("rmse-test", *rmse_test, np.mean(rmse_test), np.std(rmse_test)))
    f.write(row_format.format("r2s-test", *r2s_test, np.mean(r2s_test), np.std(r2s_test)))
    f.write(row_format.format("r2s-train", *r2s_train, np.mean(r2s_train), np.std(r2s_train)))

    for i in range(54):
        f_name = f"Feature {i}"
        arr = [weights[0][i][0], weights[1][i][0], weights[2][i][0], weights[3][i][0], weights[4][i][0]]
        f.write(row_format.format(f"Feature {i}", *arr, np.mean(arr), np.std(arr)))

    f.close()