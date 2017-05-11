# -*- coding: UTF-8 -*-
import numpy as np

class PocketPLA():
    def test(self, X):
        X_ = np.append(X, np.ones((len(X), 1)), axis=1)
        y_ = np.sign(np.inner(self.W, X_))
        return y_

    def train(self, X, y, T=5):
        X_ = np.append(X, np.ones((len(X), 1)), axis=1)
        best_W = np.zeros(X_.shape[1])
        best_accuracy = 0

        self.W = best_W
        for t in range(T):
            for x_, y_ in zip(X_, y):
                if np.sign(np.inner(self.W, x_)) != y_:
                    self.W = self.W + y_ * x_

            accuracy = np.average(y == self.test(X))
            if accuracy > best_accuracy:
                best_accuracy, best_W = accuracy, self.W

        self.W = best_W
