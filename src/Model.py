import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from joblib import Parallel, delayed
from time import time

folder = "../Collected_Data/FormattedEEG/"
types = ["control", "left_blink", "right_blink"]
paths = np.array([["left_blink_var1", 1], ["left_blink_var2", 1], ["right_blink_var1", 2], ["right_blink_var2", 2], ["control_var1", 0], ["control_var2", 0]])

# Read in csv files
def read_csvs(folder, types, paths):
    output_x = []
    output_y = []
    
    for path in paths:
        for i in range(1, 11):
            if path[0].find("control") != -1 and i == 6:
                break;
            cur_path = folder + types[int(path[1])] + "/" + path[0] + "_" + str(i) + "_" + "formatted.csv"
            input_data = pd.read_csv(cur_path, header = 0, usecols=[1, 2, 3, 4])
            X = input_data.to_numpy()[250:1000]
            X = X.T
            y = path[1]
            output_x.append(X)
            output_y.append(float(y))
    
    output_x = np.array(output_x) 
    output_y = np.array(output_y)
    return output_x, output_y

X, y = read_csvs(folder, types, paths)

def train_model(X, y, n_splits, model):
    cv = KFold(n_splits, shuffle=True)
    clf = make_pipeline(XdawnCovariances(estimator="oas"), TangentSpace(metric = "riemann"), model)
    pred = np.zeros(len(y))
    
    def train_single_set(train_idx, test_idx):
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X[train_idx], y_train)
        y_predict = clf.predict(X[test_idx])
        return dict(test_idx = test_idx, y_predict = y_predict)
        
    out = Parallel(n_jobs = 3)(delayed(train_single_set)(train_idx, test_idx) for train_idx, test_idx in cv.split(X))
    for d in out:
        pred[d["test_idx"]] = d["y_predict"]
    
    clf = clf.fit(X, y)
    
    return clf, pred

dur = time()
clf, pred = train_model(X, y, 8, RandomForestClassifier())
dur = time() - dur

print(np.mean(pred == y))
print(dur)
