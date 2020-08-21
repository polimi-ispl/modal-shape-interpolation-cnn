"""
Addition of phase to datasets of modeshapes
In dataset_small_images_phase phases were added to the starting dataset, where modes are not present the same amount
of time. Besides, both t and phi were varied, which may lead to have some images in larger measure
In dataset_small_images_phase84 all modes are present the same amount of time (84 images for each mode).
Besides, the whole argument was varied.
It is better to used this second
"""

import numpy as np
import math
import pickle


# Dataset with unbalanced mode shape images

with open("./dataset_small_images", 'rb') as data:
    dataset_base = pickle.load(data)

dataset = []

for i in range(len(dataset_base)):
    for phi in np.arange(0, 2*np.pi, 0.5):
        for t in np.arange(0, 1, 0.1):
            shape = dataset_base[i][2]*math.cos(dataset_base[0][1]*2*np.pi*t + phi)
            dataset.append((dataset_base[i][0], dataset_base[i][1], shape, dataset_base[i][3],
                            dataset_base[i][4]))

with open("./dataset_small_images_phase", 'wb') as output:
    pickle.dump(dataset, output)


# Dataset with balanced mode shape images

with open("./dataset_small_images84", 'rb') as data:
    dataset_base2 = pickle.load(data)

dataset2 = []
for i in range(len(dataset_base2)):
    for phi in np.arange(-np.pi/2, np.pi/2, 0.01):
        shape = dataset_base2[i][2]*math.sin(phi)
        dataset2.append((dataset_base2[i][0], dataset_base2[i][1], shape, dataset_base2[i][3], dataset_base2[i][4]))

with open("./dataset_small_images_phase84", 'wb') as output:
    pickle.dump(dataset2, output)
