import numpy as np
from scipy.ndimage import gaussian_filter1d


def clamp(x, min_val, max_val):
    """Clamps the given value between min and max values"""
    return max(min(x, max_val), min_val)


def periodic_corr(x, y):
    """Circular correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y).conj()).real


def conv_circ(signal, ker):
    """
    Circular convolution of a signal with a kernel.

    signal: real 1D array
    ker: real 1D array
    signal and ker must have same shape
    """
    return np.real(np.fft.ifft(np.fft.fft(signal)*np.fft.fft(ker)))


def phase_alignment(x, y, blur_width=0):
    """
    Measures the phase alignment of two vectors of the same length, by comparing their dot products with the maximum
    and minimum values of their correlation. A value of 1 means that the dot product is the highest of all possible
    rotations of one vector against the other. A value of 0 means that it is the lowest. If the maximum and minimum
    values are the same, then phase is irrelevant and 0.5 is returned.

    :param x: input vector 1
    :param y: input vector 2
    :param blur_width: standard deviation of a gaussian blur applied to y, allowing for fuzzy measurement of phase
        correlation.
    :return: value between 0 and 1
    """
    if blur_width > 0:
        y = np.array(y, dtype=float)
        y = gaussian_filter1d(y, blur_width, mode="wrap")

    correlations = periodic_corr(x, y)
    min_corr, max_corr = min(correlations), max(correlations)
    if max_corr - min_corr == 0:
        # no range, so just split it and say 0.5
        return 0.5
    else:
        return round((correlations[0] - min_corr) / (max_corr - min_corr), 10)


def window_fit_score(x, window_min, window_max, half_life=0.5):
    if window_min <= x <= window_max:
        return 1
    elif x < window_min:
        return 0.5 ** ((window_min - x) / half_life)
    else:
        return 0.5 ** ((x - window_max) / half_life)


def target_fit_score(x, target, half_life=0.5):
    return window_fit_score(x, target, target, half_life)