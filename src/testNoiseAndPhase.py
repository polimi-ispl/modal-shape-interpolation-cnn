"""
Testing of super-resolution architectures (factor 2 on each spatial dimension)
trained on dataset with varying phase and addition of noise
(each image has a random phase and snr between 20 and 80 dB)
The two tested architectures are:

simpler autoencoder which is implemented in modelsUnetNoEdges (uNet5f2)

MaxReNet which is implemented in modelsUnetNoEdges (uNet5Stackb)

Remember that the input of both networks are the interpolated images
"""

import pickle
import plotRes as pR
import math
from sklearn.metrics import mean_squared_error
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from datasetPreparation import downsample2
import numpy as np
import tensorflow as tf
from datasetReshapingUtils import cut_edges_output
from matplotlib import pyplot as plt
from modelsUnetNoEdges import uNet5Stackb, uNet5f2c



# Enter dataset file location
with open("./dataset_small_images84", 'rb') as data:
    dataset_base = pickle.load(data)

print("Length of initial dataset:", str(len(dataset_base)))

# Each row in the dataset is a tuple containing:
# (mode order (m,n), modal frequency, modeshape of dimension (8, 12), Lx, Ly))

# Addition of noise and phase
# To each image, we add noise randomly choosing a value of SNR between 80 and 20 dB
# Clean image is also added to the dataset (as last element in each tuple)

dataset = []
for i in range(len(dataset_base)):
    snr_db = np.random.uniform(20, 80)
    phase = np.random.uniform(-np.pi / 2, np.pi / 2)
    img = dataset_base[i][2]
    img = img * math.sin(phase)
    power = np.mean(img ** 2)
    var = power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(var), dataset_base[i][2].shape)
    im = dataset_base[i][2] + noise
    dataset.append((dataset_base[i][0], dataset_base[i][1], im, dataset_base[i][3], dataset_base[i][4],
                    dataset_base[i][2]))

print("Length noisy dataset: ", str(len(dataset)))

print(len(dataset), "images in dataset")

# Plot of one image for comparision
random_idx = np.random.choice(range(len(dataset)))
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(dataset[random_idx][2], cmap='coolwarm')
plt.title("Noisy image")
plt.subplot(1, 2, 2)
plt.imshow(dataset[random_idx][5], cmap='coolwarm')
plt.title("Original image")

np.random.shuffle(dataset)

# Groundtruth: original clean image without edges
groundtruth = [(dataset[ix][0], dataset[ix][1], cut_edges_output(dataset[ix][2]), dataset[ix][3], dataset[ix][4],
                cut_edges_output(dataset[ix][5])) for ix in range(len(dataset))]

# Input: downsampled groundtruth (antialiased)
downsampled = [(dataset[i][0], dataset[i][1], downsample2(groundtruth[i][2], 2), dataset[i][3], dataset[i][4],
                groundtruth[i][5]) for i in range(len(dataset))]

# Reshaping of images
groundtruth = np.array(groundtruth)
downsampled = np.array(downsampled)

groundtruth_img = np.array([groundtruth_item[5].reshape(6, 10, 1) for groundtruth_item in groundtruth])
downsampled_img = np.array([downsampled_item[2].reshape(3, 5, 1) for downsampled_item in downsampled])

# Interpolation with tensorflow
interpolated_tens = tf.image.resize_images(downsampled_img, size=[6, 10], method=ResizeMethod.BICUBIC)
# It is a tensor: turn it into np.array:
tf.InteractiveSession()
interpolated_tf = interpolated_tens.eval()



# MaxReNet

# Enter location of trained model
uNet5Stackb.load_weights('./weights_best_uNet5Stackb_noiseandphase')
uNet5Stackb.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mean_squared_error'])

score = uNet5Stackb.evaluate(interpolated_tf, groundtruth_img, verbose=0)
probs = uNet5Stackb.predict(interpolated_tf, verbose=0)

# Computing and plotting mse and nmse
mse = [mean_squared_error(groundtruth_img[idx][:, :, 0], probs[idx][:, :, 0]) for idx in
       range(len(groundtruth))]
mse_den = [mean_squared_error(groundtruth_img[idx][:, :, 0], np.zeros(groundtruth_img[idx][:, :, 0].shape))
           for idx in range(len(groundtruth))]
nmse = [mse[idx] / mse_den[idx] for idx in range(len(groundtruth))]

mse_int = [mean_squared_error(groundtruth_img[idx][:, :, 0], interpolated_tf[idx][:, :, 0]) for idx in
           range(len(groundtruth))]
nmse_int = [mse_int[idx] / mse_den[idx] for idx in range(len(groundtruth))]

pR.plot_mse(mse, mse_int)
plt.title("Mse of uNetStackb")

# Plot a random image
pR.plot_hr(probs[0][:, :, 0], groundtruth_img[0][:, :, 0], interpolated_tf[0][:, :, 0],
           downsampled_img[0][:, :, 0])
plt.title("Random reconstruction by uNetStackb")

pR.plot_mse(nmse, nmse_int)
plt.title('Normalized mean squared error uNetStackb')

# Plotting worst reconstructed image
max_mse = max(mse)
idx_bad = mse.index(max_mse)
pR.plot_hr(probs[idx_bad][:, :, 0], groundtruth_img[idx_bad][:, :, 0], interpolated_tf[idx_bad][:, :, 0],
           downsampled_img[idx_bad][:, :, 0])
plt.title("Worst reconstruction by uNetStackb")

idx_hmsebis = [index for index, value in enumerate(mse) if value > 10 ** (-10 / 10)]
modes_bad = [downsampled[idx][0] for idx in idx_hmsebis]
print("uNetStackb - score: ", str(score[0]), "; modes with mse > -10 dB:", str(len(modes_bad)))
