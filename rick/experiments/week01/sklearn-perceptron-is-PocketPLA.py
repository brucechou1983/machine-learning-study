from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500,
                           n_features=2,
                           n_informative=1,
                           n_redundant=0,
                           n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = Perceptron()
model.fit(X_train, y_train)
print accuracy_score(y_train, model.predict(X_train))
print accuracy_score(y_test, model.predict(X_test))
