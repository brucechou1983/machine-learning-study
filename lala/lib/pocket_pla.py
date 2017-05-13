import os
import json
import random
import numpy as np

class PocketPLA:
    __W = None
    def __self__():
        random.seed()

    def train(self, X, Y, I = 100):
        # add x0
        X = np.append(X, np.ones((len(X), 1)), axis = 1)

        # init weight vector
        W = np.zeros(X.shape[1])
        #
        best_W = W
        best_rate = 0.0

        for i in range(I):
            pick = int(random.random() * len(X)) % len(X)
            x = X[pick]
            y = Y[pick]
            if np.sign(np.dot(W, x)) != y:
                W = W + x * y

            rate = np.average(np.sign(np.inner(W, X)) == Y)
            if rate > best_rate:
                best_W = W
                best_rate = rate

        # setup W
        self.__W = best_W
        return self.__W

    def run(self, X):
        # add x0
        X = np.append(X, np.ones((len(X), 1)), axis = 1)
        return np.sign(np.inner(self.__W, X))


def _get_data(filename):
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../data/',
        filename
    )
    return json.load(open(filepath))

def _main():
    traindata = _get_data('not_linear_separable/train_data.json');
    traindata_X = np.array(traindata['x'])
    traindata_Y = np.array(traindata['y'])
    traindata_Y[traindata_Y == 0] = -1

    testdata = _get_data('not_linear_separable/test_data.json');
    testdata_X = np.array(testdata['x'])
    testdata_Y = np.array(testdata['y'])
    testdata_Y[testdata_Y == 0] = -1

    p = PocketPLA()
    p.train(traindata_X, traindata_Y)
    print(np.average(traindata_Y == p.run(traindata_X)))
    print(np.average(testdata_Y == p.run(testdata_X)))

if __name__ == "__main__":
    _main()
