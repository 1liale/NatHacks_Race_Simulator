from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

from ahk import AHK
import threading
import time
import sys

ahk = AHK()
buffer = np.full((300, 4), sys.maxsize)
lock = threading.Lock()
clf = load('./models/mlp.joblib')
sos = butter(8, [0.5, 100], "bandpass", output = "sos", fs = 250)

# Start a stream of eeg data
def start_stream():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    while True:
        chunk,ts = inlet.pull_sample()
        chunk = np.array(chunk)[0: 4]
        with lock:
            buffer[:-1] = buffer[1:]
            buffer[-1:] = [chunk]

# Get data from buffer
def get_data():
    with lock:
        X = buffer
    filteredX = sosfilt(sos, X)
    return filteredX

# Translate left/right blink into turning left/right controls for rFactor2
def keyboard_control(y_pred):
    if ahk.key_state("esc"):
        release_all()
        print("quit")
        sys.exit()
        return
    
    # left blink controls left turn
    if int(y_pred[0]) == 1:
        if not ahk.key_state(','):
            ahk.key_up(".")
            ahk.key_down(",")
    # right blink controls right turn
    elif int(y_pred[0]) == 2:
        if not ahk.key_state('.'):
            ahk.key_up(",")
            ahk.key_down(".")

# Start steam on a seprate thread
th = threading.Thread(target = start_stream)
th.start()

# Main loop to predict left/right blink and translates to keyboard controls
while True:
    if keyboard.is_pressed("esc"):
        print("exiting program")
        break

    # gets data from sliding stream thread
    buffer = get_data()
    model_feed_data = np.array([buffer.T])
    y_pred = clf.predict(model_feed_data)
    # steers car using model's prediction
    #keyboard_control(y_pred) # keyboard control, uncomment to simulate key presses
        
print('Streams closed')
