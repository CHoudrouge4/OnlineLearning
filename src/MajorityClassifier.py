from typing import Callable, Any, Iterable

class MajorityClassifier:

    majority_class = 0 # store the current majority class
    freq = {} # frequency table for each class
    max_freq = 0 # the current maximum frequency

    def predict1(self, x):
        return self.majority_class # return the current majority class

    # this function predict for every element in x; this function is not used.
    def predict(self, l):
        return [self.predict1(x) for x in l]

    # partial fit for one element
    def partial_fit1(self, x, y: int):
        # check if y is in the frequency table, if it is not that mean we never see it before
        if y in self.freq:
            self.freq[y] = self.freq[y] + 1 # increment the frequency
        else:
            self.freq[y] = 1 # set the freuqency to one since we saw it first time.

        #check if the frequency of the current element is larger than the frequency of the max, if yes update the majority class
        if self.max_freq < self.freq[y]:
            majority_class = y

    def partial_fit(self, X: Iterable , y: Iterable):
        n, m = X.shape
        for i in range(n):
            self.partial_fit1(X[i], y[i])
