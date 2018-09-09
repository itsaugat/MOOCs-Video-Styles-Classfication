# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 21:35:59 2018

@author: sss
"""
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = load_img('data/10.jpg')  # this is a PIL image

# convert image to numpy array with shape (3, width, height)
img_arr = img_to_array(img)

# convert to numpy array with shape (1, 3, width, height)
img_arr = img_arr.reshape((1,) + img_arr.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `data/augmented` directory
i = 0
for batch in datagen.flow(
    img_arr,
    batch_size=1,
    save_to_dir='data/code',
    save_prefix='code_A',
    save_format='jpg'):
    i += 1
    if i > 9:
        break  # otherwise the generator would loop indefinitely
