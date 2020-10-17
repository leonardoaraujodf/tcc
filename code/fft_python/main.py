import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.fftpack
import csv
from time import sleep

samples = []
with open('samples.csv') as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',')
    for row in read_csv:
        samples.append(np.float64(row)[0])

number_samples = len(samples)
sampling_frequency = 869.728
sampling_time = 1/sampling_frequency
aquisition_time = number_samples*sampling_time

# window time in seconds
# this parameter can be changed by the user
window_time = 10 

#number_samples_per_window = int(number_samples*window_time/aquisition_time)
number_samples_per_window = int(window_time * sampling_frequency)

number_windows = number_samples//number_samples_per_window

# a list of time values for each window
signal_samples_windows = []
# a list of frequency response values for each window
signal_frequency_windows = []
bpm = []

time = np.linspace(0.0, window_time, number_samples_per_window)
frequency = np.linspace(0.0, sampling_frequency/2, number_samples_per_window/2)
for window in range(number_windows):
  signal_samples_windows.append(samples[window*number_samples_per_window:(window+1)*number_samples_per_window])
  s = signal_samples_windows[window]
  s -= np.mean(s)
  s = scipy.fftpack.fft(s)
  signal_frequency_windows.append(s)
  s = np.abs(np.asarray(s[: len(s) // 2]))
  bpm.append(frequency[s.argmax()] * 60)

fig, ax = plt.subplots()
for k in range(0, len(signal_frequency_windows)):
	ax.plot(frequency, 2.0/number_samples_per_window * np.abs(signal_frequency_windows[k][:number_samples_per_window//2]))
	ax.set_xlim([0, 5])
	ax.text(3, 400000, str(int(bpm[k])) + ' batimentos por minuto')
	sleep(0.5)
plt.show()
#plt.savefig('plot.png')
print(bpm)
