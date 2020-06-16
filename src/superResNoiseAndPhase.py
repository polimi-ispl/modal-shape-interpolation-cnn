"""
Training and testing of architectures for super-resolution with downsampling factor 2 (on each dimension)
The dataset has varying phase and noise, SNR value is randomly chosen between 20 and 80 dB for each image
Considered architectures:

uNet5f2 which is implemented in modelsUnetNoEdges (simpler autoencoder)
and whose model is stored in './ModelCheckpoint/weights_best_uNet5f2_noiseandphase'

uNet5Stackb which is implemented in modelsUnetNoEdges (MaxReNet)
and whose model is stored in './ModelCheckpoint/weights_best_uNet5Stack_noiseandphase'

Remember that the input of uNet5f2 and uNet5Stack are the interpolated images
"""

import pickle
import plotRes as pR
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from datasetPreparation import downsample2
import numpy as np
import tensorflow as tf
from datasetReshapingUtils import cut_edges_output
from matplotlib import pyplot as plt
from modelsUnetNoEdges import uNet5f2c, uNet5Stackb

# Instructions for using servers
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# usa gpu con più memoria libera
import GPUtil

GPU = str(GPUtil.getFirstAvailable(order='memory')[0])
# GPU = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
print('GPU selected:', GPU)

# crea sessione tensorflow
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
'''

# Prima prova: dataset con fase varia
with open("./DatasetFiles/dataset_small_images_phase84", 'rb') as data:
    dataset_base = pickle.load(data)

print("Length of initial dataset:", str(len(dataset_base)))


# Each row in the dataset is a tuple containing:
# (mode order (m,n), modal frequency, modeshape of dimension (8, 12), Lx, Ly))

# Addition of noise
# To each image, we add noise randomly choosing a value of SNR between 80 and 20 dB
# Clean image is also added to the dataset (as last element in each tuple)

dataset = []
for i in range(len(dataset_base)):
    snr_db = np.random.uniform(20, 80)
    img = dataset_base[i][2]
    power = np.mean(img ** 2)
    var = power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(var), dataset_base[i][2].shape)
    im = dataset_base[i][2] + noise
    dataset.append((dataset_base[i][0], dataset_base[i][1], im, dataset_base[i][3], dataset_base[i][4],
                    dataset_base[i][2]))

print("Length noisy dataset with modes removed: ", str(len(dataset)))

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

# Splitting into train, validation and test sets and reshaping
groundtruth_train, groundtruth_val, groundtruth_test = np.split(np.array(groundtruth), [int(.8 * len(groundtruth)),
                                                                                        int(.9 * len(groundtruth))])
downsampled_train, downsampled_val, downsampled_test = np.split(np.array(downsampled), [int(.8 * len(downsampled)),
                                                                                        int(.9 * len(downsampled))])

groundtruth_train = np.array(groundtruth_train)
groundtruth_test = np.array(groundtruth_test)
downsampled_train = np.array(downsampled_train)
downsampled_test = np.array(downsampled_test)

groundtruth_train_img = np.array([groundtruth_item[5].reshape(6, 10, 1) for groundtruth_item in groundtruth_train])
downsampled_train_img = np.array([downsampled_item[2].reshape(3, 5, 1) for downsampled_item in downsampled_train])
groundtruth_test_img = np.array([groundtruth_item[5].reshape(6, 10, 1) for groundtruth_item in groundtruth_test])
downsampled_test_img = np.array([downsampled_item[2].reshape(3, 5, 1) for downsampled_item in downsampled_test])
groundtruth_val_img = np.array([groundtruth_item[5].reshape(6, 10, 1) for groundtruth_item in groundtruth_val])
downsampled_val_img = np.array([downsampled_item[2].reshape(3, 5, 1) for downsampled_item in downsampled_val])

# Interpolation with tensorflow for comparison
interpolated_train_tens = tf.image.resize_images(downsampled_train_img, size=[6, 10], method=ResizeMethod.BICUBIC)
interpolated_test_tens = tf.image.resize_images(downsampled_test_img, size=[6, 10], method=ResizeMethod.BICUBIC)
interpolated_val_tens = tf.image.resize_images(downsampled_val_img, size=[6, 10], method=ResizeMethod.BICUBIC)
# It is a tensor: turn it into np.array:
tf.InteractiveSession()
interpolated_train_tf = interpolated_train_tens.eval()
interpolated_test_tf = interpolated_test_tens.eval()
interpolated_val_tf = interpolated_val_tens.eval()


uNet5Stackb.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mean_squared_error'])

callback = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.2),
            ModelCheckpoint(filepath='./ModelCheckpoint/weights_best_uNet5Stack_noiseandphase',
                            monitor='val_loss', verbose=1, save_best_only=True)
            ]

print("Training noisy dataset...")
history = uNet5Stackb.fit(interpolated_train_tf, groundtruth_train_img, epochs=20, verbose=1, callbacks=callback,
                          validation_data=(interpolated_val_tf, groundtruth_val_img))

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
pR.plot_hr(probs[0][:, :, 0], groundtruth_test_img[0][:, :, 0], interpolated_test_tf[0][:, :, 0],
           downsampled_test_img[0][:, :, 0])
pR.plot_mse(nmse, nmse_int)
plt.title('Normalized mean squared error')

# Plotting worst reconstructed image
max_mse = max(mse)
idx_bad = mse.index(max_mse)
pR.plot_hr(probs[idx_bad][:, :, 0], groundtruth_test_img[idx_bad][:, :, 0], interpolated_test_tf[idx_bad][:, :, 0],
           downsampled_test_img[idx_bad][:, :, 0])

# Plotting loss
val_loss = history.history['val_loss']
loss = history.history['loss']
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(val_loss)
plt.title("Validation loss")
plt.subplot(1, 2, 2)
plt.plot(loss)
plt.title("Training loss")

# Saving history
with open('./ModelCheckpoint/trainhistory_uNet5Stack_noiseandphase', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

idx_hmsebis = [index for index, value in enumerate(mse) if value > 10 ** (-10 / 10)]
modes_bad = [downsampled_test[idx][0] for idx in idx_hmsebis]
