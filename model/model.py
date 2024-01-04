import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

samplespath = 'D:/Projects/SudokuSolver/modelimg'
samples = os.listdir(samplespath)
classes = len(samples)

# Exclude the folder for digit 0
if '0' in samples:
    samples.remove('0')

images = []
labels = []

for i in samples:
    piclist = os.listdir(samplespath + '/' + i)
    for j in piclist:
        img = cv.imread(samplespath + '/' + i +'/' + j)
        img = cv.resize(img, (28, 28))
        images.append(img)
        labels.append(i)
        
images = np.array(images)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

def proc(img):
    ima = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ima = cv.equalizeHist(ima)
    ima = ima/255
    return ima

x_train = np.array(list(map(proc, x_train)))
x_test = np.array(list(map(proc, x_test)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

def changestr(l):
    aux = []
    for i in l:
        aux.append(int(i)-1)
    return aux
    
y_train = np.array(changestr(y_train), dtype=np.uint8)
y_test = np.array(changestr(y_test), dtype=np.uint8)

model = Sequential()
model.add(Conv2D(64, (5,5),
                activation='relu',
                input_shape=(28, 28, 1),
                padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu',
                padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu',
                padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))  # Update to 9 neurons for digits 1-9
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 1
epochs = 10

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs,
          shuffle=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("my_model_digits_1_9.keras")