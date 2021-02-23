import typing as T

import numpy as np

DIFFUSION_MAPS: T.Dict[str, T.Tuple] = {
    "floyd-steinberg": (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ),
    "atkinson": (
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ),
    "jarvis-judice-ninke": (
        (1, 0, 7 / 48),
        (2, 0, 5 / 48),
        (-2, 1, 3 / 48),
        (-1, 1, 5 / 48),
        (0, 1, 7 / 48),
        (1, 1, 5 / 48),
        (2, 1, 3 / 48),
        (-2, 2, 1 / 48),
        (-1, 2, 3 / 48),
        (0, 2, 5 / 48),
        (1, 2, 3 / 48),
        (2, 2, 1 / 48),
    ),
    "stucki": (
        (1, 0, 8 / 42),
        (2, 0, 4 / 42),
        (-2, 1, 2 / 42),
        (-1, 1, 4 / 42),
        (0, 1, 8 / 42),
        (1, 1, 4 / 42),
        (2, 1, 2 / 42),
        (-2, 2, 1 / 42),
        (-1, 2, 2 / 42),
        (0, 2, 4 / 42),
        (1, 2, 2 / 42),
        (2, 2, 1 / 42),
    ),
    "burkes": (
        (1, 0, 8 / 32),
        (2, 0, 4 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 8 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32),
    ),
    "sierra3": (
        (1, 0, 5 / 32),
        (2, 0, 3 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 5 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32),
        (-1, 2, 2 / 32),
        (0, 2, 3 / 32),
        (1, 2, 2 / 32),
    ),
    "sierra2": (
        (1, 0, 4 / 16),
        (2, 0, 3 / 16),
        (-2, 1, 1 / 16),
        (-1, 1, 2 / 16),
        (0, 1, 3 / 16),
        (1, 1, 2 / 16),
        (2, 1, 1 / 16),
    ),
    "sierra-2-4a": (
        (1, 0, 2 / 4),
        (-1, 1, 1 / 4),
        (0, 1, 1 / 4),
    ),
}


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


def error_diffusion(img: np.array, method: str) -> np.array:
    ni = np.array(img, np.float)

    diff_map = DIFFUSION_MAPS.get(method.lower())

    for y in range(ni.shape[0]):
        for x in range(ni.shape[1]):
            old_pixel = ni[y, x]
            new_pixel = 1.0 if old_pixel > 0.5 else 0.0
            quantization_error = old_pixel - new_pixel
            ni[y, x] = new_pixel
            for dx, dy, diffusion_coefficient in diff_map:
                xn, yn = x + dx, y + dy
                if (0 <= xn < ni.shape[1]) and (0 <= yn < ni.shape[0]):
                    ni[yn, xn] += quantization_error * diffusion_coefficient
    return 1 - ni.astype(np.uint8)
