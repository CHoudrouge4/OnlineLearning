from skmultiflow.data import DataStream
from preprocessing import *
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.evaluation import EvaluatePrequential
import numpy
import matplotlib
from sklearn.metrics import accuracy_score
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import AdditiveExpertEnsembleClassifier

def get_stream(file_name):
    #file_name = './streams/INSECTS-abrupt_balanced_norm.csv'
    data = get_data(file_name)
    print(data.shape)
    data[:, 33] = prepare_targets(data[:, 33])
    n, m = data.shape
    labels =  data[:, 33]
    print(labels)
    data = data.astype(numpy.float64)
    labels = labels.astype(numpy.int64)
    print('X')
    data = data[:, :33]
    print(data.shape)
    print(data)
    return DataStream(data=data, y=labels), n, m

def run(model, stream, n):
    n_samples = 1000
    max_samples = n
    X, Y = stream.next_sample(1000)
    print(X.shape)
    model.partial_fit(X, Y)
    accs = []
    while n_samples < max_samples and stream.has_more_samples():
        new_x, new_y = stream.next_sample(500)

        X = numpy.vstack([X, new_x])
        Y = numpy.append(Y, new_y)
        print(X.shape)
        print(Y.shape)
        X_window = X[-1000:, :]
        Y_window = Y[-1000:]

        print('Windows shape')
        print(X_window.shape)
        print(Y_window.shape)

        Y_pred = model.predict(X_window)

        print(accuracy_score(Y_window, Y_pred))
        accs.append(accuracy_score(Y_window, Y_pred))
        model.partial_fit(X_window, Y_window)
        n_samples += 1
    return accs

def main():
    file_name = './streams/INSECTS-abrupt_balanced_norm.csv'
    dwm = DynamicWeightedMajorityClassifier()
    ht = HoeffdingTreeClassifier()
    SKNN = SAMKNNClassifier()
    HAD = HoeffdingAdaptiveTreeClassifier()
    ARF = AdaptiveRandomForestClassifier()
    AEC = AdditiveExpertEnsembleClassifier()
    Stream, n, m = get_stream(file_name)

    models = [dwm, ht, SKNN, HAD, ARF, AEC]

    evaluator = EvaluatePrequential(max_samples=n, batch_size=1000, output_file='result.csv', show_plot=True, metrics=['accuracy'])
    evaluator.evaluate(stream=Stream, model=models, model_names=['dwm', 'ht', 'SKNN', 'HAD', 'ARF', 'AEC'])

if __name__ == "__main__":
    main()
