from typing import Callable, Any, Iterable

class NoChangeClassifier:

    last_class = 0

    def predict1(self, x):
        return self.last_class

    def predict(self, l):
        return [self.predict1(x) for x in l]

    def partial_fit1(self, x, y: int):
        self.last_class = y

    def partial_fit(self, X: Iterable , y: Iterable):
        n, m = X.shape
        for i in range(n):
            self.partial_fit1(X[i], y[i])
