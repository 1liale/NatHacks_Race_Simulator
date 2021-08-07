# Record and format Muse 2 data
from muselsl import record
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "left_blink_var1_100"
input_file = "./NatHacks_Race_Simulator/Collected_Data/RawEEG/left_blink/" + file + "_raw.csv"
output_file = "./NatHacks_Race_Simulator/Collected_Data/PreprocessedEEG/left_blink/" + file + "_formatted.csv"

# Read, format, and save recorded data
def format(input_file, output_file):
  df = pd.read_csv(input_file, usecols=[1,2,3,4,5,6], header=0, names=["TP9", "AF7", "AF8","TP10", "Time", "Offset"])
  df["Time"] = df["Time"] + df["Offset"] - df["Time"].iloc[0]
  del df["Offset"]
  df.to_csv(output_file)
  return df

# Plot formatted data
def plot(df):
  fig,(graph0, graph1, graph2, graph3) = plt.subplots(nrows = 6, sharex = True)
  line0, = graph0.plot(df["Time"], df["AF7"], color = "red", linestyle = "solid")
  line1, = graph1.plot(df["Time"], df["AF8"], color = "blue", linestyle = "solid")
  line2, =  graph2.plot(df["Time"], df["TP9"], color = "red", linestyle = "dashdot")
  line3, = graph3.plot(df["Time"], df["TP10"], color = "blue", linestyle = "dashdot")

  graph0.title.set_text('AF7')
  graph1.title.set_text('AF8')
  graph2.title.set_text('TP9')
  graph3.title.set_text('TP10')

  plt.xlabel('Time')
  plt.show()

# Note: an existing Muse LSL stream is required
record(10, input_file)
# Note: Recording is synchronous, so code here will not execute until the stream has been closed
print('Recording has ended')
df = format(input_file, output_file)
plot(df)
