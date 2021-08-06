from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

from ahk import AHK
import keyboard
import threading
import time
import sys

ahk = AHK()
buffer = np.full((750, 4), sys.maxsize)
lock = threading.Lock()
clf = load('./models/mlp.joblib')

types=["control", "left", "right", "jaw"]

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
    
def get_data():
    with lock:
        temp = buffer
        return temp

def keyboard_control(y_pred):
    def release_all():
        ahk.key_up("a")
        ahk.key_up("z")
        ahk.key_up(",")
        ahk.key_up(".")

    if ahk.key_state("esc"):
        release_all()
        print("quit")
        sys.exit()
        return

    # jaw clenched controls accelerate/decelerate
    if int(y_pred[0]) == 0:
        if not ahk.key_state("z"): 
            ahk.key_up("a")
            ahk.key_up(",")
            ahk.key_up(".")
            ahk.key_down("z")
    elif int(y_pred[0]) == 3:
        if not ahk.key_state("a"): 
            ahk.key_up("z")
            ahk.key_up(",")
            ahk.key_up(".")
            ahk.key_down("a") 
    # left blink controls left turn
    if int(y_pred[0]) == 1:
        if not ahk.key_state(','):
            ahk.key_up("a")
            ahk.key_up("z")
            ahk.key_up(".")
            ahk.key_down(",")
    # right blink controls right turn
    elif int(y_pred[0]) == 2:
        if not ahk.key_state('.'):
            ahk.key_up("a")
            ahk.key_up("z")
            ahk.key_up(",")
            ahk.key_down(".")
    
# Start steam
th = threading.Thread(target = start_stream)
th.start()

counter = 0
while True:
    if keyboard.is_pressed("esc"):
        print("exiting program")
        break

    # gets data from sliding stream thread
    buffer = get_data()
    model_feed_data = np.array([buffer.T])
    y_pred = clf.predict(model_feed_data)
    # steers car using model's prediction
    keyboard_control(y_pred) # keyboard control, uncomment to simulate key presses
    if counter % 1000 == 0:
        print(types[int(y_pred[0])])
    
    counter += 1
        
print('Streams closed')
            