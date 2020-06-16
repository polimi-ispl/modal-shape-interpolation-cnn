"""
Creation of dataset_small_images84, containing balanced number of mode shape images (84 for each mode)
"""

import pickle
import collections
import random


with open("./DatasetFiles/dataset_small_images", 'rb') as data:
    dataset = pickle.load(data)

modes = [mode[0] for mode in dataset]
mode_occurences = collections.Counter(modes)


for key in mode_occurences:
    diff = 84 - mode_occurences[key]

    if diff > 0:
        idx_replica = [idx for idx in range(len(dataset)) if dataset[idx][0] == key]

        for i in range(diff):
            replica = dataset[random.choice(idx_replica)]
            dataset.append(replica)

with open("./DatasetFiles/dataset_small_images84", 'wb') as output:
    pickle.dump(dataset, output)
