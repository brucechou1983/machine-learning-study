# -*- coding: UTF-8 -*-

import json
import numpy as np
from PLA import PLA
from PocketPLA import PocketPLA

train_data = json.load(open('../../bruce/data/train_data.json'))
test_data = json.load(open('../../bruce/data/test_data.json'))
X_train = np.array(train_data['x'])
y_train = np.array(train_data['y'])
X_test = np.array(test_data['x'])
y_test = np.array(test_data['y'])
y_train[y_train==0] = -1
y_test[y_test==0] = -1

model = PLA()
model.train(X_train, y_train)
print np.average(y_train == model.test(X_train))
print np.average(y_test == model.test(X_test))

model = PocketPLA()
model.train(X_train, y_train)
print np.average(y_train == model.test(X_train))
print np.average(y_test == model.test(X_test))
