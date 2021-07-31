# Record and format Muse 2 data

from muselsl import record
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "left_blink_var1_10"
input_file = "./NatHacks_Race_Simulator/Collected_Data/RawEEG/left_blink/" + file + "_raw.csv"
output_file = "./NatHacks_Race_Simulator/Collected_Data/FormattedEEG/left_blink/" + file + "_formatted.csv"

# Note: an existing Muse LSL stream is required
record(10, input_file)
# Note: Recording is synchronous, so code here will not execute until the stream has been closed
print('Recording has ended')

# Read, format, and save recorded data
df = pd.read_csv(input_file, usecols=[1,2,3,4,5,6], header=0, names=["TP9", "AF7", "AF8","TP10", "Time", "Offset"])
df["Time"] = df["Time"] + df["Offset"] - df["Time"].iloc[0]
del df["Offset"]
df.to_csv(output_file)

# Plot formatted data
fig,(graph0, graph1, graph2, graph3, graph4) = plt.subplots(nrows = 5, sharex = True)
line0, = graph0.plot(df["Time"], df["AF7"], color = "red", linestyle = "solid")
line1, = graph1.plot(df["Time"], df["AF8"], color = "blue", linestyle = "solid")
line2, =  graph2.plot(df["Time"], df["TP9"], color = "red", linestyle = "dashdot")
line3, = graph3.plot(df["Time"], df["TP10"], color = "blue", linestyle = "dashdot")
graph4.legend((line0, line1, line2, line3), ("AF7", "AF8", "TP9", "TP10"), loc='lower right')
plt.xlabel('Time')
plt.show()

