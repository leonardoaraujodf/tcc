# Este programa realiza a leitura das amostras de PPG adquiridas em repouso utilizando dois canais.
# Realiza a divisao das amostras em janelas e calcula a resposta em frequencia do sinal adquirido.
# O local em cada janela onde ocorre o valor maximo do sinal representa, na realidade, quantos
# batimentos por minuto foram lidos

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fftpack
import csv
from time import sleep

#TODO: Review this
samples = []
with open('josiane_finger_tapping.csv') as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',')
    for row in read_csv:
        samples.append(np.float64(row)[0])
#TODO: Review this

number_samples = len(samples)

# The parameters below can be changed by the user.
# Sampling frequency in Hz.
sampling_frequency = 868.85
# Window time in seconds.
window_time = 20

sampling_time = 1/sampling_frequency
aquisition_time = number_samples*sampling_time

#number_samples_per_window = int(number_samples*window_time/aquisition_time)
number_samples_per_window = int(window_time * sampling_frequency)

number_windows = number_samples//number_samples_per_window

# A list of windows.
signal_samples_windows = []

# A list of frequency response values for each window.
signal_frequency_windows = []
bpm = []

time = np.linspace(0.0, window_time, number_samples_per_window, endpoint=False)
frequency = np.linspace(0.0, int(sampling_frequency/2), number_samples_per_window//2, endpoint=False)

print('Number samples: ', number_samples)
print('Sampling frequency: ', sampling_frequency,' Hz')
print('Sampling time: ', sampling_time, ' s')
print('Aquisition time: ', aquisition_time, ' s')
print('Number windows: ', number_windows)
print('Number samples per window: ', number_samples_per_window)

for window in range(number_windows):
  signal_samples_windows.append(samples[window*number_samples_per_window:(window+1)*number_samples_per_window])
  s = signal_samples_windows[window]
  s -= np.mean(s)
  s = scipy.fftpack.fft(s)
  signal_frequency_windows.append(s)
  s = np.abs(np.asarray(s[: len(s) // 2]))
  bpm.append(frequency[s.argmax()] * 60)

print('Batimentos por minuto: ', bpm)
fig, ax = plt.subplots()
for k in range(0, len(signal_frequency_windows)):
	ax.plot(frequency, 2.0/number_samples_per_window * np.abs(signal_frequency_windows[k][:number_samples_per_window//2]),
            label='Janela '+ str(k+1) + ' - ' + str(int(bpm[k])) + ' bpm')
	ax.set_xlim([0, 5])
	# ax.text(3, 500000 - k*25000, str(int(bpm[k])) + ' batimentos por minuto')
	sleep(0.5)

ax.set_xlabel('f (Hz)')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()
#plt.savefig('plot.png')
