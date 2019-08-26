# regression for MNIST Data set
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.training_utils import multi_gpu_model

#randomseed_fix
np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


img_rows = 28
img_cols = 28

input_shape = (img_rows, img_cols, 1) #(28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)  #(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)  #(60000,28,28,1)

x_train = x_train.astype('float32') / 255. #normalization
x_test = x_test.astype('float32') / 255.  #normalization


batch_size = 128
num_classes = 10
epochs = 10



#Regression
y_train = y_train.reshape(len(x_train), 1)
y_test = y_test.reshape(len(x_test), 1)



model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model = multi_gpu_model(model, gpus=2)
model.compile(optimizer='adam', loss='mse',metrics=["accuracy"])


#hist = model.fit(x_train, y_train, validation_data=(x_test, y_test))
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,batch_size=batch_size)

plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['loss','val_loss', 'acc','val_acc'])
plt.show()

#Score
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



n = 0
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
print(model.predict(x_test[n].reshape(1, 28, 28, 1)))
