# -*- coding: UTF-8 -*-
import numpy as np

class PLA():
    def test(self, X):
        X_ = np.append(X, np.ones((len(X), 1)), axis=1)
        y_ = np.sign(np.inner(self.W, X_))
        return y_

    def train(self, X, y):
        X_ = np.append(X, np.ones((len(X), 1)), axis=1)
        self.W = np.zeros(X_.shape[1])
        while True:
            for x_, y_ in zip(X_, y):
                if np.sign(np.inner(self.W, x_)) != y_:
                    self.W = self.W + y_ * x_
                    break
            else:
                return
