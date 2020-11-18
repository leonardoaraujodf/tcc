from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def generate_conv_neural_network(x_training, x_test, y_training, y_test):

   classifier = Sequential()

   classifier.add(Dense(units = 128, activation = 'relu'))
   classifier.add(Dropout(0.2))
   classifier.add(Dense(units = 128, activation = 'relu'))
   classifier.add(Dropout(0.2))
   classifier.add(Dense(units = 10, activation = 'softmax'))

   classifier.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
   classifier.fit(predictors_training, class_training, batch_size = 128,
                  epochs = 5, validation_data = (predictors_test, class_test))

   result = classifier.evaluate(predictors_test, class_test)
