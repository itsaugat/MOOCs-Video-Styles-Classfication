# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 23:03:53 2018

@author: sss
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical  


img_width, img_height = 150, 150

cnn_update_weights_path = './models/vgg19_model_weight.h5'
cnn_update_model = './models/vgg19_model.h5'
train_data_dir = './data/training'
validation_data_dir = './data/test'
nb_train_samples = 300
nb_validation_samples = 120
epochs = 30
batch_size = 20

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('vgg19_bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('vgg19_bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    
    class_dictionary = generator.class_indices  
    print(class_dictionary)

    
    
def train_top_model():
    train_data = np.load(open('vgg19_bottleneck_features_train.npy','rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 3) + [1] * (nb_train_samples // 3) + [2] * (nb_train_samples // 3))
    train_labels = to_categorical(train_labels, num_classes=3)

    validation_data = np.load(open('vgg19_bottleneck_features_validation.npy','rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 3) + [1] * (nb_validation_samples // 3) + [2] * (nb_validation_samples // 3))
    validation_labels = to_categorical(validation_labels, num_classes=3)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(cnn_update_weights_path)
    model.save(cnn_update_model)
    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
    print("[INFO] Loss: {}".format(eval_loss))
    
    

    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_p = range(len(acc))
    
    plt.plot(epochs_p, acc, 'b', label='Training acc')
    plt.plot(epochs_p, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy - VGG19')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs_p, loss, 'b', label='Training loss')
    plt.plot(epochs_p, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss - VGG19')
    plt.legend()
    plt.figure()
    
    plt.show()
    
save_bottlebeck_features()
train_top_model()


