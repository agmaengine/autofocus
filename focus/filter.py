import numpy as np
import scipy.signal as signal


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
    # result = signal.convolve2d(image_grey, kernel, mode='same', boundary='sym')
    result[0, :] = np.zeros_like(result[0, :])
    result[-1, :] = np.zeros_like(result[-1, :])
    result[:, 0] = np.zeros_like(result[:, 0])
    result[:, -1] = np.zeros_like(result[:, -1])
    return result


# brenner variations
def brenner_y(image_grey):
    f = image_grey[:-2, :]
    f = np.pad(f, ((1, 1), (0, 0)))
    f_padded = image_grey[2:, :]
    f_padded = np.pad(f_padded, ((1, 1), (0, 0)))
    return f**2 + f_padded**2 - 2*f*f_padded


def brenner_x(image_grey):
    f = image_grey[:, :-2]
    f = np.pad(f, ((0, 0), (1, 1)))
    f_padded = image_grey[:, 2:]
    f_padded = np.pad(f_padded, ((0, 0), (1, 1)))
    return f**2 + f_padded**2 - 2*f*f_padded


def brenner_xy(image_grey):
    x = brenner_x(image_grey)
    y = brenner_y(image_grey)
    return (x + y)/2


def squared_gradient_y(image_grey):
    f = image_grey[:-1, :]
    f = np.pad(f, ((1, 0), (0, 0)))
    f_padded = image_grey[1:, :]
    f_padded = np.pad(f_padded, ((1, 0), (0, 0)))
    return f ** 2 + f_padded ** 2 - 2 * f * f_padded


def squared_gradient_x(image_grey):
    f = image_grey[:, :-1]
    f = np.pad(f, ((0, 0), (1, 0)))
    f_padded = image_grey[:, 1:]
    f_padded = np.pad(f_padded, ((0, 0), (1, 0)))
    return f**2 + f_padded**2 - 2*f*f_padded


# first order derivative operators
def different_h(image_grey):
    kernel = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0]])
    return _apply_filter(image_grey, kernel)


def different_v(image_grey):
    kernel = np.array([[0, -1, 0],
                      [0, 0, 0],
                      [0, 1, 0]])
    return _apply_filter(image_grey, kernel)


def sobel_h(image_grey):
    kernel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    return _apply_filter(image_grey, kernel)


def sobel_v(image_grey):
    kernel = np.array([[-1, -2, -1],
                      [0, 0 ,0],
                      [1, 2, 1]])
    return _apply_filter(image_grey, kernel)


def scharr_h(image_grey):
    kernel = np.array([[-3, 0, 3],
                      [-10, 0, 10],
                      [-3, 0, 3]])
    return _apply_filter(image_grey, kernel)


def scharr_v(image_grey):
    kernel = np.array([[-3, -10, -3],
                      [0, 0, 0],
                      [3, 10, 3]])
    return _apply_filter(image_grey, kernel)


def roberts_h(image_grey):
    kernel = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [-1, 0, 0]])
    return _apply_filter(image_grey, kernel)


def roberts_v(image_grey):
    kernel = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])
    return _apply_filter(image_grey, kernel)


def prewitt_h(image_grey):
    kernel = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
    return _apply_filter(image_grey, kernel)


def prewitt_v(image_grey):
    kernel = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])
    return _apply_filter(image_grey, kernel)


# second order derivative operators
def laplacian(image_grey):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return _apply_filter(image_grey, kernel)


def sobel2_h(image_grey):
    kernel = np.array([[1, 2, 1],
                      [-2, -4, -2],
                      [1, 2, 1]])
    return _apply_filter(image_grey, kernel)


def sobel2_v(image_grey):
    kernel = np.array([[1, -2, 1],
                      [2, -4, 2],
                      [1, -2, 1]])
    return _apply_filter(image_grey, kernel)


def cross_sobel(image_grey):
    kernel = np.array([[-1, 0, 1],
                      [0, 0, 0],
                      [1, 0, -1]])
    return _apply_filter(image_grey, kernel)