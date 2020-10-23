import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft

sampling_frequency = 868.85
window_time = 20
samples_ox = np.array([77, 80, 78, 82, 80, 79, 81, 82, 78])

lines = 0
with open('josiane_finger_tapping.csv') as reader:
   for line in reader:
      lines = lines + 1

number_samples = lines
samples = np.zeros(lines)

lines = 0
with open('josiane_finger_tapping.csv') as reader:
   for line in reader:
      samples[lines] = np.float64(line)
      lines = lines + 1

sampling_time = 1/sampling_frequency
aquisition_time = number_samples*sampling_time

number_samples_per_window = int(window_time * sampling_frequency)
number_windows = number_samples//number_samples_per_window

signal_samples_windows = np.zeros((number_windows, number_samples_per_window))
signal_frequency_windows = []
bpm = np.zeros(number_windows)

time = np.linspace(0.0, window_time, number_samples_per_window, endpoint=False)
frequency = np.linspace(0.0, sampling_frequency//2, number_samples_per_window//2, endpoint=False)
print(number_samples_per_window)

for k in range(number_windows):
   signal_samples_windows[k] = samples[number_samples_per_window*k:number_samples_per_window*(k+1)]
   print(signal_samples_windows[k].size)
   s = signal_samples_windows[k]
   s -= s.mean()
   s = fft(s) 
   if k == 0:
      signal_frequency_windows = np.zeros((number_windows, s.size//2))
   signal_frequency_windows[k] = np.abs(s[:s.size//2])
   bpm[k] = frequency[signal_frequency_windows[k].argmax()] * 60

bpm_time = window_time*np.linspace(0, number_windows-1, number_windows)

# This piece of code generates 9 graphs in the same
# figure.
#
# fig, axs = plt.subplots(3, 3)
# x = 0
# y = 0
# for k in range(number_windows):
#    axs[x,y].plot(frequency, 2.0/(number_samples_per_window*10**3) * signal_frequency_windows[k], label='Janela '+ str(k+1) + ' - ' + str(int(bpm[k])) + ' bpm')
#    axs[x,y].set_xlim([0, 5])
#    axs[x,y].set_xlabel('f (Hz)')
#    axs[x,y].set_ylabel('Amplitude')
#    axs[x,y].legend()
# 
#    y += 1
#    if y == 3:
#       x += 1
#       y = 0

fig, axs = plt.subplots(2,1)
print(axs.size)
for k in range(number_windows):
   axs[0].plot(frequency, 2.0/number_samples_per_window * signal_frequency_windows[k], label='Window '+ str(k+1) + ' - ' + str(int(bpm[k])) + ' bpm')

axs[1].plot(bpm_time, bpm, label='PPG')
axs[1].plot(bpm_time, samples_ox, label='Oximeter')

axs[0].set_xlim([0, 5])
axs[1].set_xlim([0, bpm_time[number_windows-1]])

axs[0].set_xlabel('f (Hz)')
axs[0].set_ylabel('Amplitude')

axs[1].set_xlabel('t (s)')
axs[1].set_ylabel('bpm')

axs[0].legend()
axs[1].legend()

axs[0].set_title('PPG frequency response', loc='left')
axs[1].set_title('PPG vs Oximeter', loc='left')
plt.show()
