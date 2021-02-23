import numpy as np


def quantize(img: np.array) -> np.array:
    return img < 0.5


def random_noise(img: np.array) -> np.array:
    return img - 0.5 < np.random.uniform(0, 1, img.shape)


def get_bayer_matrix(n: int) -> np.array:
    mat = get_bayer_index_matrix(n)
    norm = 2 ** (2 * n + 2)
    return (1 + mat) / (1 + norm)


def get_bayer_index_matrix(n: int) -> np.array:
    if n == 0:
        return np.array([[0, 2], [3, 1]], np.float)
    mat = get_bayer_index_matrix(n - 1)
    return np.bmat(
        [
            [4 * mat + 0, 4 * mat + 2],
            [4 * mat + 3, 4 * mat + 1],
        ]
    )


def bayer(img: np.array, order=2) -> np.array:
    bayer_matrix = get_bayer_matrix(order)
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    xx %= 2 ** order
    yy %= 2 ** order
    factor_threshold_matrix = bayer_matrix[yy, xx]
    return img < factor_threshold_matrix
