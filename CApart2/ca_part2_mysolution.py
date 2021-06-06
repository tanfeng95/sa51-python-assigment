import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import glob

import os 
 
image1 = 80
image2 = 60

def preprocess(x_train, y_train, x_test, y_test):

    x_train = x_train / 255
    x_test = x_test / 255
    
    #one hot encoding for the y labels 
    # Encode label
    #LabelEncoder can be used to normalize labels.
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # convert 1 dimensional array to 4-dimensional array
    # each row in y_train and y_test is one-hot encoded
    # why 4 cause there 4 answers banna ,apple ,orange , mix so there a need to one hot encoding 4 of the ans 
    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_test = tf.keras.utils.to_categorical(y_test, 4)

    return (x_train, y_train, x_test, y_test)


def run_cnn(x_train, y_train, x_test, y_test):
    epoch = 20
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image1, image2, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())    
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    model = model.fit(x_train, y_train, batch_size=2, epochs=epoch, verbose=1,validation_data=(x_test, y_test))

    score = model.model.evaluate(x_test, y_test)
    print("score =", score)
   
    
   
print('main coding')    
x_train = []
x_test = []
y_train = []
y_test = []

def classifyImage(name):
    if name[0] == "a":
        return "apple"
    elif name[0] == "b":
        return "banana"
    elif name[0] == "o":
        return "orange"
    elif name[0] == "m":
        return "mixed"
    else:
        raise Exception()

for file in glob.iglob('C:/Users/tanfe/Desktop/SA_4108 python/CA_Part2_Dataset/train/*.jpg'):
    #print(file)
    im = Image.open(file).convert('RGB')
    size = (image1,image2)
    im = im.resize(size)
    x_train.append(np.asarray(im))
    y_train.append(classifyImage(os.path.basename(file)))
    
for file in glob.iglob('C:/Users/tanfe/Desktop/SA_4108 python/CA_Part2_Dataset/test/*.jpg'):
    #print(file)
    im = Image.open(file).convert('RGB')
    size = (image1,image2)
    im = im.resize(size)
    x_test.append(np.asarray(im))
    y_test.append(classifyImage(os.path.basename(file)))

#convert to numpy array so that keras can read them
#before asarray there no shape in the list 
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print('x_train.shape = ',x_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)
print('y_train.shape = ',y_train.shape)

(x_train, y_train, x_test, y_test) = preprocess(x_train, y_train,x_test, y_test)

model =  run_cnn(x_train, y_train, x_test, y_test)



