import numpy as np
import scipy.signal as signal
from . import filter


def _pad_kernel(image, kernel):
    h, w = image.shape[0], image.shape[1]
    kh, kw = kernel.shape[0], kernel.shape[1]
    h_pad_l = int(np.floor((h-kh)/2))
    w_pad_l = int(np.floor((w-kw)/2))
    if (h-kh) % 2 == 1:
        h_pad_r = h_pad_l+1
    else:
        h_pad_r = h_pad_l
    if (w-kw) % 2 == 1:
        w_pad_r = w_pad_l+1
    else:
        w_pad_r = w_pad_l
    kernel_padded = np.pad(kernel, ((h_pad_l, h_pad_r), (w_pad_l, w_pad_r)))
    return kernel_padded


def _apply_filter(image_grey, kernel):
    kernel = _pad_kernel(image_grey, kernel)
    result = signal.fftconvolve(image_grey, kernel, mode='same')
    return result


def _identity(array):
    return array


# brenner variations
def brenner_y(image_grey):
    return np.sum(filter.brenner_y(image_grey))


def brenner_x(image_grey):
    return np.sum(filter.brenner_x(image_grey))


def squared_gradient_y(image_grey):
    return np.sum(filter.squared_gradient_y(image_grey))


def squared_gradient_x(image_grey):
    return np.sum(filter.squared_gradient_x(image_grey))


# first order derivative operators
def different_h(image_grey):
    return np.sum(filter.different_h(image_grey))


def different_v(image_grey):
    return np.sum(filter.different_v(image_grey))


def sobel_h(image_grey):
    return np.sum(filter.sobel_h(image_grey))


def sobel_v(image_grey):
    return np.sum(filter.sobel_v(image_grey))


def scharr_h(image_grey):
    return np.sum(filter.scharr_h(image_grey))


def scharr_v(image_grey):
    return np.sum(filter.scharr_v(image_grey))


def roberts_h(image_grey):
    return np.sum(filter.roberts_h(image_grey))


def roberts_v(image_grey):
    return np.sum(filter.roberts_v(image_grey))


def prewitt_h(image_grey):
    return np.sum(filter.prewitt_h(image_grey))


def prewitt_v(image_grey):
    return np.sum(filter.prewitt_h(image_grey))


# second order derivative operators
def laplacian(image_grey):
    return np.sum(filter.laplacian(image_grey))


def sobel2_h(image_grey):
    return np.sum(filter.sobel2_h(image_grey))


def sobel2_v(image_grey):
    return np.sum(filter.sobel2_v(image_grey))


def cross_sobel(image_grey):
    return np.sum(filter.cross_sobel(image_grey))


# histogram based
def range_hist(image_grey):
    f = image_grey.flatten()
    h, bins = np.histogram(f, bins=255)
    return h.max() - h.min()


def entropy_hist(image_grey):
    """not yet implement"""
    return 0


def mason_green(image_grey, threshold):
    f = image_grey.flatten()
    h, bins = np.histogram(f, bins=255)
    return np.sum((bins-threshold)*h)


def mendelshon_mayall(image_grey):
    f = image_grey.flatten()
    h, bins = np.histogram(f, bins=255)
    b1 = bins[:-1]
    b2 = bins[1:]
    mid = (b1 + b2)/2
    return np.sum(mid * h)


# image statistics based
def variance(image_grey):
    return np.var(image_grey)


def normalize_variance(image_grey):
    return np.var(image_grey)/image_grey.mean()


def threshold_pixel_count(image_grey, function=_identity, threshold=150):
    """
    :return: sum(i(function(m,n), threshold))
    where i return 1 if function(m,n) < threshold, 0 otherwise
    """
    image_grey = function(image_grey)
    return np.sum(image_grey < threshold)


def threshold_content(image_grey, function=_identity, threshold=150):
    """
    :return: sum(s(function(m,n), threshold))
    where s return function(m,n) if function(m,n) >= threshold, 0 otherwise
    """
    image_grey = function(image_grey)
    mask = (image_grey >= threshold)
    return np.sum(mask * image_grey)


def power(image_grey, n, threshold=0):
    """
    :return: threshold_content with square function
    """
    return threshold_content(image_grey, function=lambda x: x**n, threshold=threshold)


# correlation measure
def vollath(image_grey):
    f = image_grey
    # f4
    # first terms
    f = f[:, :-1]
    f_padx = f[:, 1:]
    term1 = np.sum(f*f_padx)
    # second term
    f = f[:, :-2]
    f_padx = f[:, 2:]
    term2 = np.sum(f*f_padx)
    f4 = term1 - term2
    # f5
    f5 = term1 - f.size*f.mean()**2
    return f4, f5


def autocorrelation(image_grey, k):
    f = image_grey[:-k, :]
    f_padded = image_grey[:k, :]
    u = image_grey.mean()
    return (image_grey.size - k)*image_grey.var - np.sum((f-u)*(f_padded-u))