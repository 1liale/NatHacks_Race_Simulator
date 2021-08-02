# Project Title: EEG Race Simulator

## Objective:
Use a portable EEG device Muse 2 to steer a car in the racing simulator “rFactor 2”.

## Approach:
Using the Muse 2 and the **_muselsl_** Python module, 4 channels (AF7, AF8, TP9, TP10) of 6-second data are recorded while performing the following actions: left eye blink, right eye blink, jaw clench, and control (natural state). 65 segments were recorded for left blink, 80 for right, 60 for jaw, and 60 for control. These segments are trimmed to epochs of 3 seconds using the numpy and pandas modules which translated into 750 data points per channel given Muse’s sample rate of 250 hertz.

The dimensionality of the epochs are then reduced using **_Xdawn Covariance_** and **_Tangent Space_** operations from the **_pyRiemann_** module. The resulting data is fed into 
3 ML algorithms from the scikit-learn Python module: **_Support Vector Classifier (SVC)_**, **_Random Forest Classifier_**, and **_Multi-layer Perceptrons_**. Using cross-validation and **_Grid Search_**, the optimal model and hyperparameters are found.

The optimal model is applied to classify EEG signals in real-time. To achieve this, the LSL stream from the device is passed into a 3-second buffer and fed into the model. 
The buffer is updated on a concurrent thread and fetched by the model upon request. Once a prediction has been decided by the model, 
it is passed to a keyboard controller. The controller simulated keypresses using the wrapper module AHK, thus allowing the user to steer the car in-game.
