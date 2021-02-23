import shutil
import typing as T
from pathlib import Path

from brailleimg.conversion import img_to_braille
from brailleimg.dither import quantize, random_noise
from brailleimg.util import fit_inside

import click
import click_pathlib
import skimage.io
import skimage.transform
import skimage.util

DITHER_ALGORITHMS = {
    "quantize": quantize,
    "random-noise": random_noise,
}


@click.command()
@click.argument("path", type=click_pathlib.Path(exists=True, dir_okay=False))
@click.option(
    "--width", type=int, help="maximum output width (in terminal columns)"
)
@click.option(
    "--height", type=int, help="maximum output height (in terminal lines)"
)
@click.option(
    "--threshold", default=0.5, type=float, help='output "brightness"'
)
@click.option(
    "--dither",
    type=click.Choice(DITHER_ALGORITHMS.keys()),
    default=list(DITHER_ALGORITHMS.keys())[0],
    help="algorithm to quantize",
)
@click.option(
    "--invert",
    is_flag=True,
    help="output is meant to be viewed on a dark terminal",
)
@click.option(
    "--font-ar", default=0.5, type=float, help="adjust font aspect ratio"
)
def cli(
    width: T.Optional[int],
    height: T.Optional[int],
    threshold: float,
    dither: str,
    invert: bool,
    font_ar: float,
    path: Path,
) -> None:
    """Convert image to braille."""
    if width is None:
        width = shutil.get_terminal_size().columns - 1
    if height is None:
        height = shutil.get_terminal_size().lines - 1
    bbox_width = width << 1
    bbox_height = height << 2

    img = skimage.io.imread(path, as_gray=True)
    width, height = img.shape[1], img.shape[0]
    width, height = fit_inside(
        width * (font_ar / 0.5),  # 4:2 in Braille characters
        height,
        bbox_width,
        bbox_height,
    )

    img = skimage.transform.resize(img, (height, width), mode="constant")
    if invert:
        threshold = 1 - threshold
        img = skimage.util.invert(img)

    img = img + 0.5 - threshold
    img = DITHER_ALGORITHMS[dither](img)

    print(img_to_braille(img), end="")
