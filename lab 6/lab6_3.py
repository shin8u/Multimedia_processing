from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255  
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Flatten())  
model.add(Dense(128, activation='relu'))  

model.add(Dense(10, activation='softmax'))  

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test)) 

score = model.evaluate(x_test, y_test, verbose=0)  

model.save("my_nerone_set.keras")

print('Потеря тестовых данных: ', score[0])
print('Точность на тестовых данных: ', round(score[1], 3))
