# Usage: python expose.py <path/to/input/image.hdr> <factor> <path/to/output.{hdr,png}>

import sys

import imageio
import numpy as np


def imread(path):
    return imageio.v2.imread(path, format="HDR-FI")


exposed = imread(sys.argv[1]) * float(sys.argv[2])

output = sys.argv[3]

if output.endswith(".hdr"):
    imageio.imwrite(output, exposed, format="HDR-FI")
elif output.endswith(".png"):
    exposed = np.clip(exposed, 0, 1)
    exposed = exposed ** (1 / 2.2)
    exposed *= 255
    imageio.imwrite(output, np.rint(exposed).astype(np.uint8))
else:
    print("output format unknown.")
    exit(1)
