import csv
import os
import numpy as np
from scipy import ndimage

lines = []
data_dir = "/opt/carnd_p3/data"
with open(os.path.join(data_dir, "driving_log.csv")) as phile:
    lines = list(csv.reader(phile))[1:]

center_image_paths = list(map(lambda l: os.path.join(data_dir, "IMG", os.path.basename(l[0])), lines))
X_train = np.array(list(map(
    lambda i: ndimage.imread(i),
    center_image_paths
)))
y_train = np.array(list(map(
    lambda l: float(l[3]),
    lines
)))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

input_shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda i: (i - 128) / 256., input_shape=input_shape))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    shuffle=True,
    nb_epoch=3,
)
model.save("model.h5")
