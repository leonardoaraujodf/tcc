#!/usr/bin/env python3

# --- Imported libraries ---------------------------------------------------{{{
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
# ---}}}

def get_ppg_info():
   ppg_info = []
   ppg_info.append(
      {
         'filename' : 'signals/participanteC_coleta001.txt',
         'rest_beginning_seconds' : 4 * 60 + 20, # 4 min 20 s
         'rest_end_seconds' : 7 * 60 + 20, # 7 min 20 s
         'finger_beginning_seconds' : 60, # 1 min
         'finger_end_seconds' : 4 * 60, # 4 min
      }
   )

   ppg_info.append(
      {
         'filename' : 'signals/participanteC_coleta002.txt',
         'rest_beginning_seconds' : 4 * 60 + 10, # 4 min 10 s
         'rest_end_seconds' : 7 * 60 + 10, # 7 min 10 s
         'finger_beginning_seconds' : 60, # 1 min
         'finger_end_seconds' : 4 * 60, # 4 min
      }
   )

   ppg_info.append(
      {
         'filename' : 'signals/participanteD_coleta001.txt',
         'rest_beginning_seconds' : 30, # 30 s
         'rest_end_seconds' : 3 * 60 + 30, # 3 min 30 s
         'finger_beginning_seconds' : 3 * 60 + 45, # 3 min 45 s
         'finger_end_seconds' : 6 * 60 + 45, # 6 min 45 s
      }
   )

   ppg_info.append(
      {
         'filename' : 'signals/participanteD_coleta002.txt',
         'rest_beginning_seconds' : 20, # 20 s
         'rest_end_seconds' : 3 * 60 + 20, # 3 min 20 s
         'finger_beginning_seconds' : 3 * 60 + 30, # 3 min 30 s
         'finger_end_seconds' : 6 * 60 + 30, # 6 min 30 s
      }
   )

   ppg_info.append(
      {
         'filename' : 'signals/participanteE_coleta001.txt',
         'rest_beginning_seconds' : 60, # 1 min
         'rest_end_seconds' : 4 * 60, # 4 min
         'finger_beginning_seconds' : 4 * 60 + 20, # 4 min 20 s
         'finger_end_seconds' : 7 * 60 + 20, # 7 min 20 s
      }
   )

   ppg_info.append(
      {
         'filename' : 'signals/participanteE_coleta002.txt',
         'rest_beginning_seconds' : 60, # 1 min
         'rest_end_seconds' : 4 * 60, # 4 min
         'finger_beginning_seconds' : 4 * 60 + 10, # 4 min 10 s
         'finger_end_seconds' : 7 * 60 + 10, # 7 min 10 s
      }
   )

   return ppg_info

def ppg_extract_data():
   ppg_info = get_ppg_info()
   ppg_data = []
   for i, dic in enumerate(ppg_info):
      ppg_temp = PpgData(dic['filename'], dic['rest_beginning_seconds'], \
                    dic['rest_end_seconds'], dic['finger_beginning_seconds'], \
                    dic['finger_end_seconds'])
      ppg_temp.ppg_extraction()
      if ppg_data == []:
         ppg_data = ppg_temp
      else:
         ppg_data.ppg_vstack(ppg_temp)
   
   ppg_data.generate_time_axis()

   return ppg_data

def plot_ppg_data_in_time(ppg_data):
   import matplotlib
   matplotlib.use('TkAgg')

   print(ppg_data.x.shape)
   print(ppg_data.x_rest.shape)
   print(ppg_data.x_finger.shape)
   plt.plot(ppg_data.t_rest, ppg_data.x_rest[:, 0])
   plt.xlabel('Tempo $t$ (seconds)')
   plt.ylabel('$x_c(t)$ (volts)')
   plt.title('Channel N. 0 (rest)')
   plt.show()
   plt.plot(ppg_data.t_finger, ppg_data.x_finger[:, 0])
   plt.xlabel('Tempo $t$ (seconds)')
   plt.ylabel('$x_c(t)$ (volts)')
   plt.title('Channel N. 0 (rest)')
   plt.show()
    
class PpgData:
   def __init__(self, filename, rest_beginning_seconds, rest_end_seconds, finger_beginning_seconds, \
                finger_end_seconds, n_detectors = 5, n_sources = 2, total_fs = 4500, ad_converter_voltage = 3.3, \
                ad_converter_resolution = 2**23):
      self.filename = filename
      self.rest_beginning_seconds = rest_beginning_seconds
      self.rest_end_seconds = rest_end_seconds
      self.finger_beginning_seconds = finger_beginning_seconds
      self.finger_end_seconds = finger_end_seconds
      self.n_detectors = n_detectors
      self.n_sources = n_sources
      self.total_fs = total_fs
      self.ad_converter_voltage = ad_converter_voltage
      self.ad_converter_resolution = ad_converter_resolution
      self.x_rest = []
      self.x_finger = []
      self.t_rest = []
      self.t_finger = []
      self.x = []
      self.t_total = []
      self.n_channels = []
      self.fs = []

   def ppg_extraction(self):
      self.n_channels = self.n_sources * self.n_detectors
      with open(self.filename, 'r') as fp:
         hexvals = fp.readlines()

      decvals = [int(val.split()[0], 0) for val in hexvals]
      decvals = np.array(decvals)
      raw_signal = (self.ad_converter_voltage/self.ad_converter_resolution) * decvals
      raw_signal = raw_signal - np.mean(raw_signal)
      m = len(raw_signal) % self.n_channels
      if m != 0:
         raw_signal = raw_signal[:-m]

# Only generate a time array which has the same lenght of the signal.
# When we finish processing all data, then use generate_time_axis to really generate the time array.
      self.t_total = np.arange(len(raw_signal))

      N = int(len(raw_signal) / self.n_channels)
      self.x = raw_signal.reshape(N, self.n_channels)
      self.fs = self.total_fs / self.n_channels
      self.x_rest = deepcopy(self.x[int(self.rest_beginning_seconds * self.fs) : int(self.rest_end_seconds * self.fs), :])
      self.x_finger = deepcopy(self.x[int(self.finger_beginning_seconds * self.fs) : int(self.finger_end_seconds * self.fs), :])

   def generate_time_axis(self):
      self.t_total = (1/self.total_fs) * self.t_total
      self.t_rest = self.n_channels * deepcopy(self.t_total[int(self.rest_beginning_seconds * self.fs) : int(self.rest_end_seconds * self.fs)])
      self.t_finger = self.n_channels * deepcopy(self.t_total[int(self.finger_beginning_seconds * self.fs) : \
                                 int(self.finger_end_seconds * self.fs)])

   def adjust_time_samples(self, ppg_data):
      self.t_total = np.hstack((self.t_total, ppg_data.t_total))
      self.rest_end_seconds += ppg_data.rest_end_seconds - ppg_data.rest_beginning_seconds
      self.finger_end_seconds += ppg_data.finger_end_seconds - ppg_data.finger_beginning_seconds

   def ppg_vstack(self, ppg_data):
      self.x_rest = np.vstack((self.x_rest, ppg_data.x_rest))
      self.x_finger = np.vstack((self.x_finger, ppg_data.x_finger))
      self.x = np.vstack((self.x, ppg_data.x))
      self.adjust_time_samples(ppg_data)

'''
# --- ppg_extraction -------------------------------------------------------{{{
def ppg_extraction(filename, rest_beginning_seconds, rest_end_seconds, finger_beginning_seconds, finger_end_seconds, n_detectors = 5, n_sources = 2, total_fs = 4500.0, ad_converter_voltage = 3.3, ad_converter_resolution = 2**23):
    n_channels = n_sources * n_detectors
    with open(filename, 'r') as fp:
        hexvals = fp.readlines()
    decvals = [int(val.split()[0], 0) for val in hexvals]
    decvals = np.array(decvals)
    raw_signal = (ad_converter_voltage/ad_converter_resolution) * decvals
    raw_signal = raw_signal - np.mean(raw_signal)
    raw_signal = raw_signal
    m = len(raw_signal) % n_channels
    if m != 0:
        raw_signal = raw_signal[:-m]
    t_total = np.arange(len(raw_signal))
    t_total = (1/total_fs) * t_total
    N = int(len(raw_signal) / n_channels)
    x = raw_signal.reshape(N, n_channels)
    fs = total_fs / n_channels
    x_rest = deepcopy(x[int(rest_beginning_seconds * fs) : int(rest_end_seconds * fs), :])
    x_finger = deepcopy(x[int(finger_beginning_seconds * fs) : int(finger_end_seconds * fs), :])
    t_rest = n_channels * deepcopy(t_total[int(rest_beginning_seconds * fs) : int(rest_end_seconds * fs)])
    t_finger = n_channels * deepcopy(t_total[int(finger_beginning_seconds * fs) : \
    int(finger_end_seconds * fs)]) 
    return x_rest, x_finger, t_rest, t_finger, x, t_total
# ---}}}

# --- test -----------------------------------------------------------------{{{
if __name__ == '__main__':
    filename = 'signals/participanteC_coleta002.txt'
    rest_beginning_seconds = 4 * 60 + 10
    rest_end_seconds = rest_beginning_seconds + 3 * 60
    finger_beginning_seconds = 60
    finger_end_seconds = finger_beginning_seconds + 3 * 60
    x_rest, x_finger, t_rest, t_finger, x, t_total = \
    ppg_extraction(filename, rest_beginning_seconds, rest_end_seconds, \
    finger_beginning_seconds, finger_end_seconds)
    print(x.shape)
    print(x_rest.shape)
    print(x_finger.shape)
    plt.plot(t_rest, x_rest[:, 0])
    plt.xlabel('Tempo $t$ (seconds)')
    plt.ylabel('$x_c(t)$ (volts)')
    plt.title('Channel N. 0 (rest)')
    plt.show()
    plt.plot(t_finger, x_finger[:, 0])
    plt.xlabel('Tempo $t$ (seconds)')
    plt.ylabel('$x_c(t)$ (volts)')
    plt.title('Channel N. 0 (rest)')
    plt.show()
# ---}}}
'''
