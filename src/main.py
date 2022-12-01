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
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from statistics import mean

# get the data stream from the file called file_name
def get_stream(file_name):
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

    #adwin = HDDM_A()
    adwin = ADWIN(delta=0.0002)
    #adwin = DDM()
    alt = HoeffdingTreeClassifier()

    # pretraiing
    X, Y = stream.next_sample(1000)
    print(X.shape)
    model.partial_fit(X, Y)
    predicted_label = model.predict(X)

    changes = [0] # stores the changes point by the drift detector.
    for i in range(1000):
        adwin.add_element(int(Y[i]))
        if adwin.detected_change():
            changes.append(i)
    ##################

    accs = []
    switch_model = False
    while n_samples < max_samples and stream.has_more_samples():

        new_x, new_y = stream.next_sample() # take a sumple

        adwin.add_element(int(new_y))
        if adwin.detected_change():
            changes.append(n_samples)
            switch_model = True


        X = numpy.vstack([X, new_x]) # append the sample to X
        Y = numpy.append(Y, new_y)   # append the lable

        # getting the last 1000 labels
        Y_window = Y[-1000:]

        # getteing the prediction for the new instance.
        y_pred = model.predict(new_x) # it returns a numpy array with one element

        # saving the result of the prediction
        predicted_label = numpy.append(predicted_label, y_pred[0])

        # getting the accuracy of the last one 1000 elements
        accs.append(accuracy_score(Y_window, predicted_label[-1000:]))
        # training with class label
        model.partial_fit(new_x, new_y)
        n_samples += 1

        if switch_model:
            X_window = X[-1000:, :]
            print(X_window.shape)
            alt.partial_fit(X_window, Y_window)
            model = alt
            switch_model = False


    changes.append(max_samples)
    return accs, changes


def plotting(accs_changes, names, colors):
    #print(accs)
    fig, ax = plt.subplots()
    #for ((acc, name), c) in zip(zip(accs, names), colors):
    #    ax.plot(acc, c, label=name)
    accs = []
    changes = []
    for i in range(len(accs_changes)):
        acc, change = accs_changes[i]
        accs.append(acc)
        changes.append(change)

    for (acc, name) in zip(accs, names):
        print(name)
        print(mean(acc))

    for i in range(len(accs)):
        ax.plot(accs[i], color=colors[i], label=names[i])
        for j in range(len(changes[i]) - 1):
            if j % 2 == 0:
                ax.fill_between(changes[i][j:j+2], 0, 1, color='green', alpha=0.5, transform=ax.get_xaxis_transform())
            else:
                ax.fill_between(changes[i][j:j+2], 0, 1, color='red', alpha=0.5, transform=ax.get_xaxis_transform())
    ax.set(xlabel='Number of Points', ylabel='Accuracy')
    leg = ax.legend();
    plt.show()

#experiment function, calls run and do the plotting funciton
def experiment_1(models, names, colors, stream, n):
    accs_changes = [run(m, stream, n) for m in models]
    print(len(accs_changes))
    plotting(accs_changes, names, colors)

# the main method
def main():
    #file_name = './streams/INSECTS-abrupt_balanced_norm.csv'
    #file_name = './streams/INSECTS-gradual_balanced_norm.csv'
    file_name = './streams/INSECTS-incremental_balanced_norm.csv'
    mc = MajorityClassifier()
    nc = NoChangeClassifier()
    ht = HoeffdingTreeClassifier()
    SKNN = SAMKNNClassifier(n_neighbors=5)
    HAD = HoeffdingAdaptiveTreeClassifier()
    ARF = AdaptiveRandomForestClassifier()
    AEC = AdditiveExpertEnsembleClassifier()
    stream, n, m = get_stream(file_name)

    #models = [mc, nc , ht, SKNN, HAD, ARF, AEC]
    models = [ht]
    #models = [mc, nc]
    #names = ['MC', 'NC', 'HT', 'SKNN', 'HAD', 'ARF', 'AEC']
    names = ['HT']
    #colors = ['r', 'b', 'g', 'y', 'm', 'b', 'c']
    colors = ['m']
    experiment_1(models, names, colors, stream, n)

if __name__ == "__main__":
    main()
