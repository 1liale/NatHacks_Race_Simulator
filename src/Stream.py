from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import numpy as np

import keyboard
import threading
import time

buffer = np.zeros((750, 4))
lock = threading.lock()

def start_stream():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    while True:
        chunk,_ = inlet.pull_sample()
        chunk = np.array(chunk)[0: 4]
        with lock:
            buffer[:-1] = buffer[1:]
            buffer[-1:] = [chunk]

def get_data():
    with lock:
        return buffer

# Start steam
th = threading.Thread(target = start_steam)

while True:
    if keyboard.is_pressed("esc"):
        print("exiting program")
        break

    buffer = get_data()
    print(buffer.shape)

print('Streams closed')
