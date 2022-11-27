from typing import Callable, Any, Iterable

class MajorityClassifier:

    majority_class = 0
    freq = {}
    max_freq = 0

    def predict1(self, x):
        return self.majority_class

    def predict(self, l):
        return [self.predict1(x) for x in l]

    def partial_fit1(self, x, y: int):
        if y in self.freq:
            self.freq[y] = self.freq[y] + 1
            if self.max_freq < self.freq[y]:
                majority_class = y
        else:
            self.freq[y] = 1
            if self.max_freq < self.freq[y]:
                majority_class = y

    def partial_fit(self, X: Iterable , y: Iterable):
        n, m = X.shape
        for i in range(n):
            self.partial_fit1(X[i], y[i])
