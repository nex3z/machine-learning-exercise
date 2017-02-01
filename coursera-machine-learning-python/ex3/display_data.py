import matplotlib.pyplot as plt
import numpy as np

from format_tile import format_tile


def display_data(x, tile_width=-1, padding=0, show=False):
    """
    Display data in a nice grid

    Parameters
    ----------
    x : ndarray
        Raw data.
    tile_width : int
        Width of each image.
    padding : int
        Padding around the image.
    show : bool
        True to show the plot immediately.
    """
    m, n = x.shape

    if tile_width < 0:
        tile_width = int(np.round(np.sqrt(n)))
    tile_height = n / tile_width

    display_rows = int(np.floor(np.sqrt(m)))
    display_columns = int(np.ceil(m / display_rows))

    tile_height_padded = tile_height + padding * 2
    tile_width_padded = tile_width + padding * 2
    data = np.zeros((display_rows * tile_height_padded, display_columns * tile_width_padded))

    for i in range(display_rows):
        for j in range(display_columns):
            tile = format_tile(x[i * display_rows + j, ], tile_width, padding)
            tile = tile.T
            data[i * tile_height_padded:(i + 1) * tile_height_padded,
                 j * tile_width_padded:(j + 1) * tile_width_padded] = tile

    plt.imshow(data, cmap='gray', extent=[0, 1, 0, 1])

    if show:
        plt.show()
