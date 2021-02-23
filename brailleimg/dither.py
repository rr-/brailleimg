import numpy as np


def quantize(img: np.array) -> np.array:
    return img < 0.5


def random_noise(img: np.array) -> np.array:
    return img - 0.5 < np.random.uniform(0, 1, img.shape)
