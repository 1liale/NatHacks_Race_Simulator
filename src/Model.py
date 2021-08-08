import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from scipy.signal import butter, sosfilt

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from joblib import dump, load

folder = "../Collected_Data/FormattedEEG/"
types = ["control", "left_blink", "right_blink"]
paths = np.array([["left_blink_var1", 1], ["right_blink_var1", 2], ["control_var1", 0]])

# Grid search params
RandomForest_params = {"xdawncovariances__nfilter": [1, 2, 3, 4, 5, 6, 7, 8],
                       "randomforestclassifier__n_estimators": [500, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100],
                       "randomforestclassifier__criterion": ["gini", "entropy"],
                       "randomforestclassifier__max_features": [None, "sqrt", "log2"]}

SVC_params = {"xdawncovariances__nfilter": [1, 2, 3, 4, 5, 6, 7, 8],
              "svc__C": [0.1, 0.5, 1, 5, 10, 50, 100],
              "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1, 10, 100] ,
              "svc__kernel": ["rbf"],
              "svc__decision_function_shape": ["ovo"]}

MLP_params = {"xdawncovariances__nfilter": [1, 2, 3, 4, 5, 6, 7, 8],
              "mlpclassifier__hidden_layer_sizes": [(150,), (175,), (200,), (225,), (250,), (275,), (300,)],
              "mlpclassifier__activation": ["relu", "tanh", "logistic"],
              "mlpclassifier__solver": ["adam", "sgd", "lbfgs"],
              "mlpclassifier__learning_rate": ["constant", "invscaling", "adaptive"],
	      "mlpclassifier__max_iter": [10000]}

# Read and preprocess in csv files
def read_csvs(folder, types, paths):
    output_x = []
    output_y = []
    sos = butter(8, [0.5, 100], "bandpass", output = "sos", fs = 250)
    
    for path in paths:
        i = 1
        while True:
            try:
                cur_path = folder + types[int(path[1])] + "/" + path[0] + "_" + str(i) + "_" + "formatted.csv"
                input_data = pd.read_csv(cur_path, header = 0, usecols=[1, 2, 3, 4])
                input = input_data.to_numpy()
                Xs = [input[200:500], input[500:800], input[800:1100]]
                y = int(path[1])
                for X in Xs:
                    filteredX = sosfilt(sos, X)
                    output_x.append(filteredX.T)
                    output_y.append(y)
                i += 1
            except Exception:
                break
    
    output_x = np.array(output_x) 
    output_y = np.array(output_y)
    return output_x, output_y

# Train model
def train_model(X, y, model, cv, params):
    optimized_model = GridSearchCV(model, params, scoring= "balanced_accuracy", n_jobs = -1, refit = True, cv = cv)
    optimized_model.fit(X, y)
    print(optimized_model.best_params_)
    return optimized_model

# Find accuracy score and run time
def evaluate_model(X, y, model, cv):
    scores = cross_validate(model, X, y, cv = cv, n_jobs = -1)
    print(model)
    print(f"Fit time: {np.average(scores['fit_time'])}")
    print(f"Score time: {np.average(scores['score_time'])}")
    print(f"Test score: {np.average(scores['test_score'])}")
    return scores

# Read in data and setup cross validation
X, y = read_csvs(folder, types, paths)
print("Finished reading")
cv = KFold(10, shuffle=True)
"""
# Setup piplines
randomForest_Pipepline = make_pipeline(XdawnCovariances(nfilter = 4, estimator = "oas"), TangentSpace(metric = "riemann"), RandomForestClassifier())
sVC_Pipepline = make_pipeline(XdawnCovariances(nfilter = 4, estimator = "oas"), TangentSpace(metric = "riemann"), SVC())
mLP_Pipepline = make_pipeline(XdawnCovariances(nfilter = 4, estimator = "oas"), TangentSpace(metric = "riemann"), MLPClassifier(max_iter = 1000))

# Find optimal params
optimal_RandomForest = train_model(X, y, randomForest_Pipepline, cv, RandomForest_params)
optimal_SVC = train_model(X, y, sVC_Pipepline, cv, SVC_params)
optimal_MLP = train_model(X, y, mLP_Pipepline, cv, MLP_params)
"""

# The best params are found and new pipeline with those params are created
optimal_RandomForest_Pipeline = make_pipeline(XdawnCovariances(nfilter = 2, estimator = "oas"), TangentSpace(metric = "riemann"), RandomForestClassifier(criterion = "entropy", max_features = "sqrt", n_estimators = 800))
optimal_SVC_Pipeline = make_pipeline(XdawnCovariances(nfilter = 4, estimator = "oas"), TangentSpace(metric = "riemann"), SVC(C = 50, decision_function_shape = "ovo", gamma = 0.001, kernel = "rbf",))
optimal_MLP_Pipeline = make_pipeline(XdawnCovariances(nfilter = 4, estimator = "oas"), TangentSpace(metric = "riemann"), MLPClassifier(activation = "relu", hidden_layer_sizes = (250,), learning_rate = "constant", solver = "sgd", max_iter = 10000))

# Evaluate models
evaluate_model(X, y, optimal_RandomForest_Pipeline, cv)
evaluate_model(X, y, optimal_SVC_Pipeline, cv)
evaluate_model(X, y, optimal_MLP_Pipeline, cv)

# Dump trained models
dump(optimal_RandomForest_Pipeline.fit(X, y), './models/' + "RandomForest.joblib")
dump(optimal_SVC_Pipeline.fit(X, y), './models/' + "SVC.joblib")
dump(optimal_MLP_Pipeline.fit(X, y), './models/' + "MLP.joblib")

