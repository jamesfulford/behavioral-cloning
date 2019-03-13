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

center_image_paths = list(map(lambda l: os.path.join(data_dir, "IMG", os.path.basename(l[0])), lines))
center_images = list(map(ndimage.imread, center_image_paths))

output_values = list(map(
    lambda l: float(l[3]),
    lines
))

#
# Building training sets
#
X_train = np.array(
    center_images  # original data
    + list(map(np.fliplr, center_images))  # horizontal flip
)

y_train = np.array(
    output_values  # original data
    + list(map(lambda y: -y, output_values))  # horizontal flip
)

from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Lambda, Convolution2D, pooling

input_shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda i: (i - 128) / 256., input_shape=input_shape))

model.add(Convolution2D(20, 5, 5))
model.add(Activation("relu"))
model.add(pooling.MaxPooling2D())

model.add(Convolution2D(15, 5, 5))
model.add(Activation("relu"))
model.add(pooling.MaxPooling2D())
                     
model.add(Flatten())

model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(100))
model.add(Activation("relu"))

model.add(Dense(20))
model.add(Activation("relu"))

model.add(Dense(5))
model.add(Activation("relu"))

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
