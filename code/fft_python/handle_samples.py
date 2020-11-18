import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import get_window
import matplotlib.pyplot as plt

def get_samples_from_file(filename):
   """
   This function receives a filename and returns the number of samples and the samples that are in it in numpy
   float64.

   Parameters:
      filename: the file which has the samples

   Returns:
      samples: A numpy float64 array which has the samples
      number_samples: The number of samples in the file
   """
   lines = open(filename).readlines()
   number_samples = len(lines)
   samples = np.zeros(number_samples)

   for index, line in enumerate(lines):
      samples[index] = np.float64(line)

   return number_samples, samples

def get_next_power2(number):
   i = number
   next_power = 1
   while( i >= 1 ):
      next_power *= 2
      i //= 2

   return next_power

def compute_windowing(samples):
   hamming = get_window('hamming', samples)
   n = get_next_power2(samples)

   return hamming, n
   

def convert_samples_to_processed_windows(samples, number_samples, sampling_frequency, time_for_each_window_s):
   """
   This function process the received samples using the sampling_frequency, divides the samples
   using the given time of each window and generates a frequency response for each separated for
   each of these windows. A hamming window is used to process the frequency response samples generated.

   Parameters:
      samples: A numpy64 array which has the samples to process.
      number_samples: The number of samples in the samples array.
      sampling_frequency: The sampling frequency used to generate the samples array.
      time_for_each_windows_s: The time in seconds choosen to split the samples array.

   Returns:
     number_windows: The number of windows (a integer value) for the samples array.
     windows_frequency: A array which has in each index a array of frequency response
     values calculated based in the samples array.
     frequency: A array which has the frequency axis.
     bpm: Another array which has the beats per minutes generated using the samples
     array.
   """
   sampling_time = 1 / sampling_frequency
   aquisition_time = number_samples * sampling_time

   number_samples_per_window = int(time_for_each_window_s * sampling_frequency)
   number_windows = number_samples // number_samples_per_window

# windows time is the variable which has all the individual windows. This is a matrix which has
# number_samples_per_window columns and number_windows lines.
   windows_time = np.zeros((number_windows, number_samples_per_window))

# windows frequency is a matrix which has the same characteristic, but for frequency response.
   windows_frequency = []

# bpm - beats per minute
   bpm = np.zeros(number_windows)

   time = np.linspace(0.0, time_for_each_window_s, number_samples_per_window, endpoint = False)
   frequency = np.linspace(0.0, sampling_frequency // 2, number_samples_per_window // 2, endpoint = False)

   hamming, fft_number_samples = compute_windowing('hamming', number_samples_per_window)
   print(number_samples_per_window)

   for i in range(number_windows):
      windows_time[i] = samples[(number_samples_per_window * i):(number_samples_per_window * (i + 1))]
      actual_window_time = windows_time[i]

# Remove the mean to improve the frequency response
      actual_window_time -= actual_window_time.mean()
      actual_window_freq = fft(actual_window_time)
      actual_window_freq_size = actual_window_freq.size // 2
      if i == 0:
         windows_frequency = np.zeros((number_windows, actual_window_freq_size))

# Just uses half of the frequency response, because the other side has the same 
# information mirrored. Also, we are handling with complex numbers, so just get
# the magnitude values for us.
      actual_window_freq_abs = np.abs(actual_window_freq[:actual_window_freq_size])

# This is a frequency window, store it.
      windows_frequency[i] = actual_window_freq_abs

# Get the location (in the array) which occurs the maximum value (magnitude value)
      loc_max_value_frequency = actual_window_freq_abs.argmax()

# Convert this to bpm
      bpm[i] = frequency[loc_max_value_frequency] * 60
   
   return number_windows, number_samples_per_window, windows_frequency, frequency, bpm
