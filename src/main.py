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
from MajorityClassifier import *
from NoChangeClassifier import *
import matplotlib.pyplot as plt

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
    if not stream.has_more_samples():
        print('NO more samples!!!!')
        stream.restart()
        print('Restarting...')
    print('Start a run')
    n_samples = 1000
    max_samples = n

    # pretraiing
    X, Y = stream.next_sample(1000)
    print(X.shape)
    model.partial_fit(X, Y)
    predicted_label = model.predict(X)
    ##################

    accs = []
    while n_samples < max_samples and stream.has_more_samples():
        new_x, new_y = stream.next_sample() # take a sumple

        X = numpy.vstack([X, new_x]) # append the sample to X
        Y = numpy.append(Y, new_y)   # append the lable

        # getting the last 1000 labels
        Y_window = Y[-1000:]

        # getteing the prediction for the new instance.
        y_pred = model.predict(new_x) # it returns a numpy array with one element
        #print(y_pred)

        # saving the result of the prediction
        predicted_label = numpy.append(predicted_label, y_pred[0])

        # getting the accuracy of the last one 1000 elements
        accs.append(accuracy_score(Y_window, predicted_label[-1000:]))
        # training with class label
        model.partial_fit(new_x, new_y)
        n_samples += 1
    return accs


def plotting(accs, names, colors):
    #print(accs)
    fig, ax = plt.subplots()
    #for ((acc, name), c) in zip(zip(accs, names), colors):
    #    ax.plot(acc, c, label=name)
    for i in range(len(accs)):
        ax.plot(accs[i], color=colors[i], label=names[i])
    ax.set(xlabel='Number of Points', ylabel='Accuracy')
    leg = ax.legend();
    plt.show()

def experiment_1(models, names, colors, stream, n):
    accs = [run(m, stream, n) for m in models]
    print(len(accs))
    plotting(accs, names, colors)

def main():
    file_name = './streams/INSECTS-abrupt_balanced_norm.csv'
    #file_name = './streams/INSECTS-gradual_balanced_norm.csv'
    #file_name = './streams/INSECTS-incremental_balanced_norm.csv'
    mc = MajorityClassifier()
    nc = NoChangeClassifier()
    ht = HoeffdingTreeClassifier()
    SKNN = SAMKNNClassifier(n_neighbors=5)
    HAD = HoeffdingAdaptiveTreeClassifier()
    ARF = AdaptiveRandomForestClassifier()
    AEC = AdditiveExpertEnsembleClassifier()
    stream, n, m = get_stream(file_name)

    models = [mc, nc , ht, SKNN, HAD, ARF, AEC]
    #models = [mc, nc]
    names = ['MC', 'NC', 'HT', 'SKNN', 'HAD', 'ARF', 'AEC']
    colors = ['r', 'b', 'g', 'y', 'm', 'b', 'c']
    experiment_1(models, names, colors, stream, n)

    # evaluator = EvaluatePrequential(max_samples=n, batch_size=1000, output_file='result.csv', show_plot=True, metrics=['accuracy'])
    # evaluator.evaluate(stream=Stream, model=models, model_names=['dwm', 'ht', 'SKNN', 'HAD', 'ARF', 'AEC'])

if __name__ == "__main__":
    main()
