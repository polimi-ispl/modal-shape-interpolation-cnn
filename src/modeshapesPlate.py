"""
Functions for the analytic expressions of rectangular plates (mode frequencies and mode shapes),
the computation of sampling step and of all the mode shapes satisfying sampling condition and maximum temporal frequency
"""

import math
import numpy as np
from matplotlib import pyplot as plt


def fmode(m, n, Lx, Ly, C, th):
    """
    Returns the frequency in Hz of mode (m,n) for a rectangular plate
    :param m: mode order along x dimension
    :param n: mode order along y dimension
    :param Lx: length of plate along x dimension
    :param Ly: length of plate along y dimension
    :param C: speed of propagation
    :param th: thickness of plate
    :return: the temporal frequency in Hz for the mode (m, n) for a plate with given dimensions
    """
    return 0.453 * C * th * ((m / Lx) ** 2 + (n / Ly) ** 2)


def cl(Eym, Rho, Nu):
    """
    Returns the velocity of wave
    :param Eym: Young modulus
    :param Rho: density of material
    :param Nu: Poisson coefficient
    :return: propagation velocity
    """
    return math.sqrt(Eym / (Rho * (1 - Nu ** 2)))


def z(x, y, m, n, Lx, Ly, A):
    """
    Returns the mode shape of mode (m,n)
    :param x: x axis
    :param y: y axis
    :param m: mode order along x dimension
    :param n: mode order along y dimension
    :param Lx: length of plate along x dimension
    :param Ly: length of plate along y dimension
    :param A: amplitude of mode shape (can be set to 1)
    :return: the function of x and y describing the mode shape (displacement along z direction)
    """
    return np.sin(m * math.pi * x / Lx) * np.sin(n * math.pi * y / Ly)


def create_grid(Lx, Ly, d):
    """
    Creates a grid of points given the two dimensions of the
    rectangular plate and the distance between two samples
    :param Lx: length of plate along x dimension
    :param Ly: length of plate along y dimension
    :param d: sampling distance between two consecutive points
    :return: mesh of points
    """
    x = np.arange(0, Lx + 0.0001, d)
    y = np.arange(0, Ly + 0.0001, d)
    x, y = np.meshgrid(x, y)
    return x, y


def plot_shape(shape):
    """
    Plots the given mode shape as a 2D image
    :param shape: the mode shape to plot
    :return:
    """
    plt.figure()
    plt.imshow(shape, cmap='coolwarm', origin='lower')
    plt.xlabel("[cm]")
    plt.ylabel("[cm]")
    plt.colorbar()


def modes_shapes(f_max, Lx, Ly, th, d, Eym, Rho, Nu):
    """
    Given a plate's dimensions, finds all the modes that are below the given maximum temporal frequency and whose
    spatial frequency satisfies the sampling theorem for the given delta.
    Stores the mode order, its temporal freqeuncy and its mode shape
    :param f_max: the maximum temporal frequency
    :param Lx: length of plate along x dimension
    :param Ly: length of plate along y dimension
    :param th: thickness of plate
    :param d: sampling step between two consecutive points
    :param Eym: Young modulus
    :param Rho: density if material
    :param Nu: Poisson coefficient
    :return: a dictionary whose keys are the couples (m,n) of mode orders and values a tuple containing
    the corresponding temporal frequency and the mode shape
    """
    C = cl(Eym, Rho, Nu)
    modes = find_modes(f_max, Lx, Ly, C, th)
    res = {}
    x, y = create_grid(Lx, Ly, d)

    for i, f in modes.items():
        (m, n) = i

        kx = m * np.pi / Lx
        ky = n * np.pi / Ly
        k_max = max(kx, ky)
        k_s = 1.2*(2*k_max)

        if k_s < 2*np.pi/d:
            shape = z(x, y, m, n, Lx, Ly, A=1)
            res[(m, n)] = (f, shape)

    return res


def find_modes(f_max, Lx, Ly, C, th):
    """
    Given the plate dimensions and a maximum frequency, finds all the modes that are below to that frequency
    :param f_max: maximum temporal frequency to reach
    :param Lx: length along x dimension
    :param Ly: length along y dimension
    :param C: speed propagation
    :param th: thickness of plate
    :return: a dictionary whose keys are the couples of mode orders (m, n)
    and values the corresponding temporal frequencies
    """
    modes = {}

    for m in range(1, 1000):
        f = fmode(m, 1, Lx, Ly, C, th)
        if f > f_max:
            break
        for n in range(1, 1000):
            f = fmode(m, n, Lx, Ly, C, th)
            if f > f_max:
                break
            else:
                modes[m, n] = f

    return modes


def delta_sampling_max(f_max, Lx, Ly, C, th):
    """
    Finds the maximum delta needed to sample all mode shapes up to a given maximum frequency,
    given the plate dimensions
    :param f_max: maximum temporal frequency in Hz to reach
    :param Lx: length along x dimension
    :param Ly: length along y dimension
    :param C: speed of propagation
    :param th: thickness of plate
    :return: delta needed to sample the mode shape with maximum frequency (therefore all mode shapes up to
    that frequency
    """
    modes = find_modes(f_max, Lx, Ly, C, th)
    f = 0
    for i in modes:
        if modes[i] > f:
            f = modes[i]
            (m, n) = i

    d = delta_sampling_mn(Lx, Ly, m, n)
    return d


def delta_sampling_mn(Lx, Ly, m, n):
    """
    Computes the maximum delta needed to sample mode shape of order (m,n) given the plate dimensions
    Modes m and n determine two spatial frequencies kx, ky along the two axes: we need to make sure that the spatial
    sampling frequency (and therefore sampling delta) satisfies the Nyquist theorem with kmax = max(kx, ky)
    :param Lx: length of plate along x dimension
    :param Ly: length of plate along y dimension
    :param m: mode order along x dimension
    :param n: mode order along y dimension
    :return:
    """
    kx = m * np.pi / Lx
    ky = n * np.pi / Ly
    kmax = max(kx, ky)
    ks = 1.2 * (2 * kmax)
    d = 2 * np.pi / ks
    return d


lx = 0.8
ly = 0.8

nu = 0.3
E = 2.1e11
rho = 7850
h = 0.002
delta = 0.02

c = cl(E, rho, nu)

fmax = 2000
find = find_modes(fmax, lx, ly, c, h)
mod = modes_shapes(fmax, lx, ly, h, delta, E, rho, nu)
# to get list of modes: sorted(mod)
# to plot a certain mode shape: plot_shape(mod[m,n][1])
