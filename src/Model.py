import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import plot_confusion_matrix

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


from joblib import Parallel, delayed
from time import time

folder = "../Collected_Data/FormattedEEG/"
types = ["control", "left_blink", "right_blink", "jaw_clench"]
paths = np.array([["left_blink_var1", 1], ["left_blink_var2", 1], ["right_blink_var1", 2], ["right_blink_var2", 2], ["control_var1", 0], ["control_var2", 0], ["control_var3", 0], ["jaw_clench_var1", 3], ["jaw_clench_var2", 3]])

# Read in csv files
def read_csvs(folder, types, paths):
    output_x = []
    output_y = []
    
    for path in paths:
        i = 1
        while True:
            try:
                cur_path = folder + types[int(path[1])] + "/" + path[0] + "_" + str(i) + "_" + "formatted.csv"
                # print(cur_path)
                input_data = pd.read_csv(cur_path, header = 0, usecols=[1, 2, 3, 4])
                X = input_data.to_numpy()[250:1000]
                X = X.T
                y = path[1] 
                output_x.append(X)
                output_y.append(float(y))
                i += 1
            except:
                break;
    
    output_x = np.array(output_x) 
    output_y = np.array(output_y)
    return output_x, output_y

# Train model
def train_model(X, y, pipeline, cv, params):
    return GridSearchCV(pipeline, params, scoring= "balanced_accuracy", n_jobs = -1, refit = True, cv = cv)

# Find accuracy score
def evaluate_model(X, y, pipeline, cv):
    clf = pipeline
    pred = np.zeros(len(y))
    
    def train_single_set(train_idx, test_idx):
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X[train_idx], y_train)
        y_predict = clf.predict(X[test_idx])
        return dict(test_idx = test_idx, y_predict = y_predict)
        
    out = Parallel(n_jobs = -1)(delayed(train_single_set)(train_idx, test_idx) for train_idx, test_idx in cv.split(X))
    for d in out:
        pred[d["test_idx"]] = d["y_predict"]
    
    clf = clf.fit(X, y)
    
    return clf, pred

X, y = read_csvs(folder, types, paths)
cv = KFold(5, shuffle=True)

randomForest_Pipepline = make_pipeline(XdawnCovariances(estimator = "oas"), TangentSpace(metric = "riemann"), RandomForestClassifier())
sVC_Pipepline = make_pipeline(XdawnCovariances(estimator = "oas"), TangentSpace(metric = "riemann"), SVC())
mLP_Pipepline = make_pipeline(XdawnCovariances(estimator = "oas"), TangentSpace(metric = "riemann"), MLPClassifier())


RandomForest_params = {"xdawncovariances__nfilter": [2, 4, 8],
                       "randomforestclassifier__n_estimators": [50, 100, 200, 400, 600, 800, 1000],
                       "randomforestclassifier__criterion": ["gini", "entropy"],
                       "randomforestclassifier__max_features": [None, "sqrt", "log2"]}

optimal_RandomForest = train_model(X, y, randomForest_Pipepline, cv, RandomForest_params)
optimal_RandomForest.fit(X, y)
dump(optimal_RandomForest, './models/OptimalRandomForest.joblib')


SVC_params = {"xdawncovariances__nfilter": [2, 4, 8],
              "svc__C": [0.1, 0.5, 1, 10, 100],
              "svc__gamma": ["scale", 1, 0.01, 0.001, 0.0001] ,
              "svc__kernel": ["rbf"],
              "svc__decision_function_shape": ["ovo", "ovr"]}

optimal_SVC = train_model(X, y, sVC_Pipepline, cv, SVC_params)
optimal_SVC.fit(X, y)
dump(optimal_SVC, './models/Optimal_SVC.joblib')

         
MLP_params = {"xdawncovariances__nfilter": [2, 4, 8],
              "mlpclassifier__hidden_layer_sizes": [(100,), (300,), (500,),(100, 100), (300, 300), (500, 500)],
              "mlpclassifier__activation": ["relu", "tanh"],
              "mlpclassifier__solver": ["adam", "sgd", "lbfgs"],
              "mlpclassifier__batch_size": [100, 200, 250],
              "mlpclassifier__learning_rate": ["constant", "invscaling", "adaptive"]}

optimal_MLP = train_model(X, y, mLP_Pipepline, cv, MLP_params)
optimal_MLP.fit(X, y)
dump(optimal_MLP, './models/Optimal_MLP.joblib')

""" 
dur = time()
clf, pred = train_model(X, y, pipeline, cv)
dur = time() - dur

acc = np.mean(pred == y)
                        
print(f"Test cases: {len(y)}")
print(f"Classification accuracy: {acc}")
print(f"Time taken: {dur}" )
plot_confusion_matrix(clf, X, y, display_labels=types)
plt.show() """
