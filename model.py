import csv
import os
from math import ceil

import numpy as np
from scipy import ndimage
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    Cropping2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
)

#
# Configurations
#
data_dir = "/opt/carnd_p3/data"
input_shape = (160, 320, 3)  # pixels, 3 channels (RGB)

# Adjust based on computer's performance characteristics (higher if better)
batch_size = 48

# Degrees to correct left/right images by.
correction = .2

# Percentage to set aside for validation set
validation_set_size = .2

# Top of image is mostly noise, bottom includes hood. Crop top 70, bottom 25.
cropping = (70, 25)

# Number of epochs to undergo during training
epochs = 10


#
# Reading lines from driving_log.csv
#
lines = []
with open(os.path.join(data_dir, "driving_log.csv")) as phile:
    lines = list(csv.reader(phile))[1:]

train_samples, validation_samples = train_test_split(
    lines,
    test_size=validation_set_size
)


#
# Define generators to access data
#
def get_data_from_line(line, correction=.2):
    """
    Returns images and values for given row in .csv.
    Data augments include left-to-right flips of each image.
    """
    # Read each image
    images = list(map(
        lambda i: ndimage.imread(
            os.path.join(data_dir, "IMG", os.path.basename(line[i]))
        ),
        # images recorded from center camera are in the first column,
        # from left camera are in the second column,
        # from right camera are in the third column,
        range(3)  # center: 0, left: 1, right: 2
    ))
    value = float(line[3])  # 4th column holds steering angle
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

            # Prepare a batch
            batch_images, batch_values = [], []
            for sample_line in batch_samples:
                images, values = get_data_from_line(
                    sample_line,
                    correction=correction
                )
                # include all returned images and values in batch
                batch_images.extend(images)
                batch_values.extend(values)

            yield sklearn.utils.shuffle(
                np.array(batch_images),
                np.array(batch_values),
            )


#
# Define model
#
model = Sequential()
model.add(Cropping2D(cropping=(cropping, (0, 0)), input_shape=input_shape))
model.add(Lambda(lambda i: (i - 128) / 256.))  # Mean centering + normalization
# (subtract integer first before making things floats => better)

model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))  # output layer


#
# Train model
#
model.compile(loss="mse", optimizer="adam")
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
    epochs=epochs,
)
model.save("model.h5")
