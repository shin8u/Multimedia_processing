import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = Sequential()

model.add(Dense(512, input_shape=(784, ), activation='relu'))
model.add(Dense(768, activation='relu'))

model.add(Dense(10, activation='softmax'))


optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=3, batch_size=128, callbacks=[tensorboard_callback])

accuracy = model.evaluate(x_test,y_test)
print(f'Точность {accuracy[1]*100:.2f}%')

model.save("my_model.keras")
