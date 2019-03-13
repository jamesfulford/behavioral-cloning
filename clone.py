import csv
import os
from math import ceil

import numpy as np
from scipy import ndimage
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Dropout, Lambda, Convolution2D
from keras.layers.pooling import MaxPooling2D

#
# Reading data
#
lines = []
data_dir = "/opt/carnd_p3/data"
with open(os.path.join(data_dir, "driving_log.csv")) as phile:
    lines = list(csv.reader(phile))[1:]

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


#
# Define generators to access data
#
def get_data_from_line(line, correction=.2):
    images = list(map(
        lambda i: ndimage.imread(
            os.path.join(data_dir, "IMG", os.path.basename(line[i]))
        ),
        range(3)  # center: 0, left: 1, right: 2
    ))
    value = float(line[3])
    values = [value, value + correction, value - correction]
    return (
        images +  # center, left, and right camera images
        list(map(np.fliplr, images)),  # flipped images
        values +  # center, left, and right angles
        list(map(lambda y: -y, values)),  # flipped angles
    )


def data_generator(samples, batch_size=128, correction=.2):
    n = len(samples)
    while True:
        samples = sklearn.utils.shuffle(samples)
        for i in range(0, n, batch_size):
            batch_samples = samples[i:i + batch_size]

            batch_images, batch_values = [], []
            for sample_line in batch_samples:
                images, values = get_data_from_line(
                    sample_line,
                    correction=correction
                )
                batch_images.extend(images)
                batch_values.extend(values)

            yield sklearn.utils.shuffle(
                np.array(batch_images),
                np.array(batch_values),
            )


#
# Define model
#
input_shape = (160, 320, 3)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
model.add(Lambda(lambda i: (i - 128) / 256.))

model.add(Convolution2D(12, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(400))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(25))
model.add(Dropout(0.5))
model.add(Dense(1))


#
# Train model
#
model.compile(loss="mse", optimizer="adam")

batch_size = 32
correction = .2
model.fit_generator(
    data_generator(
        train_samples,
        batch_size=batch_size,
        correction=correction
    ),
    steps_per_epoch=ceil(len(train_samples) / batch_size),
    validation_data=data_generator(
        validation_samples,
        batch_size=batch_size,
        correction=correction
    ),
    validation_steps=ceil(len(validation_samples) / batch_size),
    epochs=10,
)
model.save("model.h5")
