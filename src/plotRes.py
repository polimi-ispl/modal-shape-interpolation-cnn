"""
Functions to plot the results of mode shape images super-resolution
"""

from matplotlib import pyplot as plt
import numpy as np


def plot_hr(rec, gt, interp, down):
    """
    Shows the reconstructed mode shape images together with the groundtruth, the interpolation and the downsampled input
    :param rec: reconstructed image
    :param gt: groundtruth
    :param interp: reconstruction by interpolation
    :param down: downsampled input
    """
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(rec, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    plt.xlabel("[cm]")
    plt.ylabel("[cm]")
    plt.title("Reconstructed image")

    plt.subplot(2, 2, 2)
    plt.imshow(gt, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    plt.xlabel("[cm]")
    plt.ylabel("[cm]")
    plt.title("Original image")

    plt.subplot(2, 2, 3)
    plt.imshow(interp, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    plt.xlabel("[cm]")
    plt.ylabel("[cm]")
    plt.title("Interpolated image")

    plt.subplot(2, 2, 4)
    plt.imshow(down, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    plt.xlabel("[cm]")
    plt.ylabel("[cm]")
    plt.title("Downsampled image")

    plt.colorbar()


def plot_mse(mse_rec, mse_interp):
    """
    Plots the mse in dB between the reconstruction and the groundtruth, compared with the mse between the interpolation
    and the groundtruth
    :param mse_rec: mse for each sample between reconstruction and groundtruth
    :param mse_interp: mse for each sample between intepolation and groundtruth
    """
    mse_rec = 10*np.log10(mse_rec)
    mse_interp = 10*np.log10(mse_interp)
    plt.figure()
    plt.plot(mse_rec)
    plt.plot(mse_interp)
    plt.show()
    plt.legend(['Reconstructed image', 'Interpolated image'])
