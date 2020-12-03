from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

def join_windows(x, y, mode = 'alternate'):

   n_windows, n_samples = x.shape
   cnn_input = np.zeros((2 * n_windows, n_samples))
   return cnn_input, clasS

def convert_windows_to_cnn_inputs(windows_rest, windows_finger, test_ratio = 0.7):
   n_windows, _ = windows_rest.shape
   n_training_input = int(test_ratio * n_windows)
   n_test_input = n_windows - n_training_input
   x_training, y_training = join_windows(windows_rest[0:n_training_input, :], \
                                             windows_finger[0:n_training_input, :])
   x_test, y_test = join_windows(windows_rest[n_training_input:, :], \
                                     windows_finger[n_training_input:, :])

   return x_training, y_training, x_test, y_test

def generate_conv_neural_network(x_training, y_training, x_test, y_test):

c = Sequential()
c.add(Conv1D(filters = 4, kernel_size = 1, activation = 'relu', input_shape = [2, 1]))
c.add(MaxPooling1D(1))
c.add(Flatten())
c.add(Dense(units = 2, activation = 'softmax'))

c.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
c.fit(predictors_training, class_training, batch_size = 128,
               epochs = 5, validation_data = (predictors_test, class_test))

