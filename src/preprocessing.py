import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, SelectFpr, GenericUnivariateSelect
import numpy as np

def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    #print(le.classes_)
    return y_enc

# read data from file.
def get_data(file_name):
    # Read the file in pandas data frame
    data = pd.read_csv(file_name, header=None)
    # store the datasfrom sklearn.metrics import accuracy_scoreet
    dataset = data.values
    return dataset
