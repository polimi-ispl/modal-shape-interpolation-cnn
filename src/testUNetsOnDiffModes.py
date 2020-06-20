import pickle
import plotRes as pR
from sklearn.metrics import mean_squared_error
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from datasetPreparation import downsample2
import numpy as np
import tensorflow as tf
from datasetReshapingUtils import cut_edges_output
from matplotlib import pyplot as plt
from modelsUnetNoEdges import uNet5f2c, uNet5Stackb

# Enter dataset location
with open("./DatasetFiles/dataset_small_images84", 'rb') as data:
    dataset_base = pickle.load(data)

print("Length of initial dataset:", str(len(dataset_base)))

# Remove images related to modes: (3, 2), (2, 4), (1, 3), (7, 2)
# Second experiment: remove images (3, 2), (1, 2), (2, 1), (4, 1)

# Each row in the dataset is a tuple containing:
# (mode order (m,n), modal frequency, modeshape of dimension (8, 12), Lx, Ly))

dataset_train = []
dataset_test = []
for i in range(len(dataset_base)):
    phase = np.random.choice([-1, 1])
    img = dataset_base[i][2]
    img = img * np.math.sin(phase)
    snr_db = np.random.uniform(20, 80)
    img = dataset_base[i][2]
    power = np.mean(img ** 2)
    var = power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(var), dataset_base[i][2].shape)
    im = dataset_base[i][2] + noise
    if ((dataset_base[i][0] != (1, 2))
            and (dataset_base[i][0] != (2, 1))
            and (dataset_base[i][0] != (4, 1))
            and (dataset_base[i][0] != (3, 2))):
        dataset_train.append((dataset_base[i][0], dataset_base[i][1], im, dataset_base[i][3], dataset_base[i][4],
                              dataset_base[i][2]))
    else:
        dataset_test.append((dataset_base[i][0], dataset_base[i][1], im, dataset_base[i][3], dataset_base[i][4],
                             dataset_base[i][2]))

print("Length noisy dataset with modes removed: ", str(len(dataset_train)))
print("Length noisy dataset of remaining modes (test): ", str(len(dataset_test)))

np.random.shuffle(dataset_train)
np.random.shuffle(dataset_test)

# Test sets
groundtruth_test = [
    (dataset_test[ix][0], dataset_test[ix][1], cut_edges_output(dataset_test[ix][2]), dataset_test[ix][3],
     dataset_test[ix][4], cut_edges_output(dataset_test[ix][5])) for ix in range(len(dataset_test))]

# Input: downsampled groundtruth (antialiased)
downsampled_test = [(dataset_test[i][0], dataset_test[i][1], downsample2(groundtruth_test[i][2], 2), dataset_test[i][3],
                     dataset_test[i][4],
                     groundtruth_test[i][5]) for i in range(len(dataset_test))]

groundtruth_test = np.array(groundtruth_test)
downsampled_test = np.array(downsampled_test)

groundtruth_test_img = np.array([groundtruth_item[5].reshape(6, 10, 1) for groundtruth_item in groundtruth_test])
downsampled_test_img = np.array([downsampled_item[2].reshape(3, 5, 1) for downsampled_item in downsampled_test])

# Interpolation with tensorflow for comparison
interpolated_test_tens = tf.image.resize_images(downsampled_test_img, size=[6, 10], method=ResizeMethod.BICUBIC)
# It is a tensor: turn it into np.array:
tf.InteractiveSession()
interpolated_test_tf = interpolated_test_tens.eval()

# Enter location of saved trained model
uNet5Stackb.load_weights('./ModelCheckpoint/weights_best_uNet5Stackb_noiseandphase_modes2')
uNet5Stackb.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mean_squared_error'])

scoreb = uNet5Stackb.evaluate(interpolated_test_tf, groundtruth_test_img, verbose=0)
probsb = uNet5Stackb.predict(interpolated_test_tf, verbose=0)

# Computing and plotting mse and nmse
mseb = [mean_squared_error(groundtruth_test_img[idx][:, :, 0], probsb[idx][:, :, 0]) for idx in
        range(len(groundtruth_test_img))]
mse_denb = [mean_squared_error(groundtruth_test_img[idx][:, :, 0], np.zeros(groundtruth_test_img[idx][:, :, 0].shape))
            for idx in range(len(groundtruth_test_img))]
nmseb = [mseb[idx] / mse_denb[idx] for idx in range(len(groundtruth_test_img))]

mse_intb = [mean_squared_error(groundtruth_test_img[idx][:, :, 0], interpolated_test_tf[idx][:, :, 0]) for idx in
            range(len(groundtruth_test_img))]
nmse_intb = [mse_intb[idx] / mse_denb[idx] for idx in range(len(groundtruth_test_img))]

pR.plot_mse(mseb, mse_intb)
plt.title("Mse uNetStackb")
pR.plot_hr(probsb[0][:, :, 0], groundtruth_test_img[0][:, :, 0], interpolated_test_tf[0][:, :, 0],
           downsampled_test_img[0][:, :, 0])
plt.title("Rec by uNetStackb")
pR.plot_mse(nmseb, nmse_intb)
plt.title('Normalized mean squared error uNetStackb')

# Plotting worst reconstructed image
max_mseb = max(mseb)
idx_badb = mseb.index(max_mseb)
pR.plot_hr(probsb[idx_badb][:, :, 0], groundtruth_test_img[idx_badb][:, :, 0], interpolated_test_tf[idx_badb][:, :, 0],
           downsampled_test_img[idx_badb][:, :, 0])

idx_hmsebisb = [index for index, value in enumerate(mseb) if value > 10 ** (-10 / 10)]
modes_badb = [downsampled_test[idx][0] for idx in idx_hmsebisb]

print("uNetStackb - score: ", str(scoreb[0]), "- modes bad:", str(len(modes_badb)))
