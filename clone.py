import csv
import os
import numpy as np
import cv2
from scipy import ndimage

#
# Reading data
#
lines = []
data_dir = "/opt/carnd_p3/data"
with open(os.path.join(data_dir, "driving_log.csv")) as phile:
    lines = list(csv.reader(phile))[1:]

image_paths = (
    list(map(lambda l: os.path.join(data_dir, "IMG", os.path.basename(l[0])), lines))  # center
    + list(map(lambda l: os.path.join(data_dir, "IMG", os.path.basename(l[1])), lines))  # left
    + list(map(lambda l: os.path.join(data_dir, "IMG", os.path.basename(l[2])), lines))  # right
)
all_images = list(map(ndimage.imread, image_paths))
center_values = list(map(lambda l: float(l[3]), lines))
correction = .2
all_values = (
    center_values  # center
    + list(map(lambda y: y + correction, center_values))  # left
    + list(map(lambda y: y - correction, center_values))  # right
)
                  
X_train = np.array(
    all_images
    + list(map(np.fliplr, all_images))
)
y_train = np.array(
    all_values
    + list(map(lambda y: -y, all_values))
)

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, pooling

input_shape = (160, 320, 3)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda i: (i - 128) / 256.))

model.add(Convolution2D(10, 5, 5, activation="relu"))
model.add(pooling.MaxPooling2D())

model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(pooling.MaxPooling2D())

model.add(Flatten())

model.add(Dense(400))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    shuffle=True,
    epochs=3,
)
model.save("model.h5")
