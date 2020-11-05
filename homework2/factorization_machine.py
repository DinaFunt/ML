import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FactorizationMachine(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, n_factors=3,
                 eta=0.01, reg_factors=0.01, random_state=1234):
        self.n_iter = n_iter
        self.n_factors = n_factors
        self.reg_factors = reg_factors
        self.random_state = random_state
        self.eta = eta

    def gradient_descent(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self._w = np.zeros(n_features)
        self._v = np.random.normal(scale=1 / np.sqrt(self.n_factors), size=(self.n_factors, n_features))

        for i in range(0, self.n_iter):
            _sgd_update(X.data, X.indptr, X.indices, y, n_samples, self._w, self._v, self.n_factors, self.eta)

    def predict(self, X):
        linear_output = X * self._w
        v = self._v.T
        term = (X * v) ** 2 - (X.power(2) * (v ** 2))
        factor_output = 0.5 * np.sum(term, axis=1)
        return linear_output + factor_output


def _sgd_update(data, indptr, indices, y, n_samples, w, v, n_factors, eta):
    # one step of sgd
    # Compute the loss of the current iteration and update gradients accordingly.

    for i in range(n_samples):
        y_one_pred, summed = _predict_instance(data, indptr, indices, w, v, n_factors, i)

        # calculate loss and its gradient
        loss_gradient = 2 * np.subtract(y_one_pred, y[i]) / len(y[i])
        # х * (y - ŷ) / N

        # update weight
        for index in range(indptr[i], indptr[i + 1]):
            feature = indices[index]
            w[feature] -= eta * loss_gradient * data[index]

        # update factor
        for factor in range(n_factors):
            for index in range(indptr[i], indptr[i + 1]):
                feature = indices[index]
                term = summed[factor] - v[factor, feature] * data[index]
                v_gradient = loss_gradient * data[index] * term
                v[factor, feature] -= eta * v_gradient


def _predict_instance(data, indptr, indices, w, v, n_factors, i):
    # Similar to _predict_instance but predicting a single instance
    # predicting y for only one sample --- y_one_pred
    # and calculating the independent term --- summed

    summed = np.zeros(n_factors)
    summed_squared = np.zeros(n_factors)

    # linear output w * x
    pred = 0
    for index in range(indptr[i], indptr[i + 1]):
        feature = indices[index]
        pred += w[feature] * data[index]

    # factor output
    for factor in range(n_factors):
        for index in range(indptr[i], indptr[i + 1]):
            feature = indices[index]
            term = v[factor, feature] * data[index]
            summed[factor] += term
            summed_squared[factor] += term * term

        pred += 0.5 * (summed[factor] * summed[factor] - summed_squared[factor])

    return pred, summed
