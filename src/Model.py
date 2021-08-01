import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import plot_confusion_matrix

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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

X, y = read_csvs(folder, types, paths)


cv = KFold(n_splits=10, shuffle=True)

# with svc
#clf = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(metric = "riemann"), SVC(C = 1.0, kernel = "rbf", gamma = "auto"))

# with random forest
clf = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(metric = "riemann"), RandomForestClassifier())

# with Neural Network
#clf = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(metric = "riemann"), MLPClassifier())

pred = np.zeros(len(y))

for train_idx, test_idx in cv.split(X):
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X[train_idx], y_train)
    pred[test_idx] = clf.predict(X[test_idx])

acc = np.mean(pred == y)
print("Test cases: " + str(len(y)))
print("Classification accuracy: %f " % (acc))
print("Time taken: " + str(end - start))
plot_confusion_matrix(clf, X, y, display_labels=types)  
plt.show()  