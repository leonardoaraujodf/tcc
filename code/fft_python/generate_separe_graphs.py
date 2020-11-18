import matplotlib
import matplotlib.pyplot as plt
from handle_samples import *

sampling_frequency = 868.85
time_for_each_window_s = 20
samples_ox = np.array([77, 80, 78, 82, 80, 79, 81, 82, 78])

# Create a function to open the file, extract the samples and return them

number_samples, samples = get_samples_from_file('josiane_finger_tapping.csv')
number_windows, number_samples_per_window, windows_frequency, frequency, bpm = \
  convert_samples_to_processed_windows(samples, number_samples, sampling_frequency, time_for_each_window_s) 

bpm_time = time_for_each_window_s*np.linspace(0, number_windows-1, number_windows)

# This piece of code generates 9 graphs in the same
# figure.
#
# fig, axs = plt.subplots(3, 3)
# x = 0
# y = 0
# for k in range(number_windows):
#    axs[x,y].plot(frequency, 2.0/(number_samples_per_window*10**3) * windows_frequency[k], label='Janela '+ str(k+1) + ' - ' + str(int(bpm[k])) + ' bpm')
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
   axs[0].plot(frequency, 2.0/number_samples_per_window * windows_frequency[k], label='Window '+ str(k+1) + ' - ' + str(int(bpm[k])) + ' bpm')

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
