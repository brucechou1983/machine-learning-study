import os
import json
import numpy as np

class PLA:
    def __init__(self):
        self.__W = None

    def train(self, X, Y):
        # add x0
        X = np.append(X, np.ones((len(X), 1)), axis = 1)

        # init weight vector
        W = np.zeros(X.shape[1])

        stop = False
        while not stop:
            stop = True
            for x, y in zip(X, Y):
                if np.sign(np.dot(W, x)) != y:
                    W = W + x * y
                    stop = False
                    break

        # setup W
        self.__W = W
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
    traindata = _get_data('linear_separable/train_data.json');
    traindata_X = np.array(traindata['x'])
    traindata_Y = np.array(traindata['y'])
    traindata_Y[traindata_Y == 0] = -1

    testdata = _get_data('linear_separable/test_data.json');
    testdata_X = np.array(testdata['x'])
    testdata_Y = np.array(testdata['y'])
    testdata_Y[testdata_Y == 0] = -1

    p = PLA()
    p.train(traindata_X, traindata_Y)
    print(np.average(traindata_Y == p.run(traindata_X)))
    print(np.average(testdata_Y == p.run(testdata_X)))

if __name__ == "__main__":
    _main()
