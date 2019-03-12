import csv
import os
import numpy as np
from scipy import ndimage

lines = []
with open("../data/driving_log.csv") as phile:
    lines = list(csv.reader(phile))

X_train = np.array(list(map(
    lambda l: ndimage.imread(os.path.join("..", "data", "IMG", os.path.basename(l[0].split("/")))),
    lines
)))
y_train = np.array(list(map(
    lambda l: float(l[3]),
    lines
)))
print(len(X_train), len(y_train))