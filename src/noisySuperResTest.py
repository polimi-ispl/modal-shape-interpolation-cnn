"""
Testing of super-resolution architecture (factor 2 on each spatial dimension)
with noisy dataset

MaxReNet (uNet5Stackb) implemented in modelsUnetNoEdges
Architectures were tested on datasets with different SNR than the one they were trained on,
in particular: train on 40 dB and test on 30 dB ('./ModelCheckpoint/weights_best_uNetStack_noiseandphase_40vs30')
train on 30 dB and test on 40 dB ('./ModelCheckpoint/weights_best_uNetStack_noiseandphase_30vs40')
"""

import pickle
import math
import numpy as np
import tensorflow as tf
import plotRes as pR
from sklearn.metrics import mean_squared_error
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from datasetPreparation import downsample2
from datasetReshapingUtils import cut_edges_output
from matplotlib import pyplot as plt
from modelsUnetNoEdges import uNet5Stackb


with open("./DatasetFiles/dataset_small_images84", 'rb') as data:
    dataset_base = pickle.load(data)

print("Length of initial dataset:", str(len(dataset_base)))


# Each row in the dataset is a tuple containing:
# (mode order (m,n), modal frequency, modeshape of dimension (8, 12), Lx, Ly))

dataset20 = dataset_base

# Addition of noise
snr_db = 30

# Addition of noise and phase variation
ds_20 = []
for i in range(len(dataset20)):
    A = np.random.uniform(-np.pi / 2, -np.pi / 5)
    B = np.random.uniform(np.pi/5, np.pi/2)
    phase = np.random.choice([A, B])
    img = dataset_base[i][2]
    img = img * math.sin(phase)
    power = np.mean(img ** 2)
    var = power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(var), dataset20[i][2].shape)
    im = dataset20[i][2] + noise
    ds_20.append((dataset20[i][0], dataset20[i][1], im, dataset20[i][3], dataset20[i][4]))

print("Length noisy dataset: ", str(len(ds_20)))

# Addition of clean image to every row of the dataset
dataset = []
for i in range(len(ds_20)):
    dataset.append((ds_20[i][0], ds_20[i][1], ds_20[i][2], ds_20[i][3], ds_20[i][4], dataset20[i][2]))

print(len(dataset), "images in dataset")

# Plot of one image for comparision
idx = np.random.choice(len(dataset))
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(dataset[idx][2], cmap='coolwarm')
plt.title("Noisy image")
plt.subplot(1, 2, 2)
plt.imshow(dataset[idx][5], cmap='coolwarm')
plt.title("Original image")

np.random.shuffle(dataset)

# Groundtruth: original clean image without edges
groundtruth = [(dataset[ix][0], dataset[ix][1], cut_edges_output(dataset[ix][2]), dataset[ix][3], dataset[ix][4],
                cut_edges_output(dataset[ix][5])) for ix in range(len(dataset))]

# Input: downsampled groundtruth (antialiased)
downsampled = [(dataset[i][0], dataset[i][1], downsample2(groundtruth[i][2], 2), dataset[i][3], dataset[i][4],
                groundtruth[i][5]) for i in range(len(dataset))]
'''
downsampled = []
for i in range(len(dataset)):
    downsampled_im = downsample2(groundtruth[i][2], 2)
    max_el = downsampled_im.max()
    min_el = downsampled_im.min()
    downsampled_im = 2 * ((downsampled_im - min_el) / (max_el - min_el)) - 1
    downsampled.append((dataset[i][0], dataset[i][1], downsampled_im, dataset[i][3], dataset[i][4],
                        groundtruth[i][5]))
'''
# Splitting into train, validation and test sets and reshaping

groundtruth_test = groundtruth
downsampled_test = downsampled

groundtruth_test_img = np.array([groundtruth_item[5].reshape(6, 10, 1) for groundtruth_item in groundtruth_test])
downsampled_test_img = np.array([downsampled_item[2].reshape(3, 5, 1) for downsampled_item in downsampled_test])

# Interpolation with tensorflow for comparison
interpolated_test_tens = tf.image.resize_images(downsampled_test_img, size=[6, 10], method=ResizeMethod.BICUBIC)
# It is a tensor: turn it into np.array:
tf.InteractiveSession()
interpolated_test_tf = interpolated_test_tens.eval()


# Test MaxReNet
# Load previously trained model
uNet5Stackb.load_weights('./ModelCheckpoint/weights_best_uNet5Stack_noiseandphase_40vs30')
uNet5Stackb.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mean_squared_error'])

score = uNet5Stackb.evaluate(interpolated_test_tf, groundtruth_test_img, verbose=0)
probs = uNet5Stackb.predict(interpolated_test_tf, verbose=0)

# Computing and plotting mse and nmse
mse = [mean_squared_error(groundtruth_test_img[idx][:, :, 0], probs[idx][:, :, 0]) for idx in
       range(len(groundtruth_test))]
mse_den = [mean_squared_error(groundtruth_test_img[idx][:, :, 0], np.zeros(groundtruth_test_img[idx][:, :, 0].shape))
           for idx in range(len(groundtruth_test))]
nmse = [mse[idx] / mse_den[idx] for idx in range(len(groundtruth_test))]

mse_int = [mean_squared_error(groundtruth_test_img[idx][:, :, 0], interpolated_test_tf[idx][:, :, 0]) for idx in
           range(len(groundtruth_test))]
nmse_int = [mse_int[idx] / mse_den[idx] for idx in range(len(groundtruth_test))]

pR.plot_mse(mse, mse_int)
plt.title('Mean squared error - uNetStackb')
pR.plot_hr(probs[0][:, :, 0], groundtruth_test_img[0][:, :, 0], interpolated_test_tf[0][:, :, 0],
           downsampled_test_img[0][:, :, 0])
plt.title("Rec by uNetStackb")

pR.plot_mse(nmse, nmse_int)
plt.title('Normalized mean squared error - uNetStackb')

# Plotting worst reconstructed image
max_mse = max(mse)
idx_bad = mse.index(max_mse)
pR.plot_hr(probs[idx_bad][:, :, 0], groundtruth_test_img[idx_bad][:, :, 0], interpolated_test_tf[idx_bad][:, :, 0],
           downsampled_test_img[idx_bad][:, :, 0])
plt.title("Worst rec by uNetStackb")

idx_hmsebis = [index for index, value in enumerate(mse) if value > 10 ** (-10 / 10)]
modes_bad = [downsampled_test[idx][0] for idx in idx_hmsebis]
print("uNetStackb - score:", str(score[0]), "modes bad:", str(len(modes_bad)))

print("Worst mode by uNetStackb: ", str(idx_bad), "-", str(downsampled[idx_bad][0]))
