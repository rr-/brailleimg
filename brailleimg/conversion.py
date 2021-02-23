import numpy

INDICES = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1)]


def img_to_braille(img):
    width, height = img.shape[1], img.shape[0]
    padded_height = ((height + 3) >> 2) << 2
    padded_width = ((width + 1) >> 1) << 1
    img = numpy.pad(
        img,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="constant",
        constant_values=1,
    )

    output = ""
    for row in numpy.vsplit(img, padded_height >> 2):
        for col in numpy.split(row, padded_width >> 1, axis=1):
            idx = 0
            for i in range(8):
                y, x = INDICES[i]
                idx |= col[y, x] << i
            output += chr(0x2800 + idx)
        output += "\n"
    return output
