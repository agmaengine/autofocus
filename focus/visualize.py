import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def show_filter(image_grey, _filter, title='', axes_handle=plt):
    filtered = _filter(image_grey)
    # if filtered.max() > np.abs(filtered.min()):
    #     lim = filtered.max()
    # else:
    #     lim = np.abs(filtered.min())
    lim = 3*filtered.std()
    lim = lim.astype('float32')
    norm = Normalize(vmax=lim, vmin=-lim)
    axes_handle.imshow(filtered, cmap='RdBu', norm=norm)
    plt.colorbar()
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def show_filter_3d(image_grey, _filter, title='', axes_handle=None):
    axes_handle = plt.axes(projection='3d')
    axes_handle.set_title(title)
    filtered = _filter(image_grey)
    y, x = filtered.shape
    y = np.arange(y)
    x = np.arange(x)
    x, y = np.meshgrid(x, y)
    axes_handle.plot_surface(x, y, filtered)
