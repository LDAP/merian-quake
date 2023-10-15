# Combine multiple HDR images into one.
# Usage: python combine_images.py path/to/input/images path/to/output.{hdr,png}

import sys
from pathlib import Path

import imageio
import numpy as np


def imread(path):
    return imageio.v2.imread(path, format="HDR-FI")


combined = np.array([imread(r) for r in Path(sys.argv[1]).iterdir() if r.suffix == ".hdr"]).mean(axis=0)

output = sys.argv[2]

if output.endswith(".hdr"):
    imageio.imwrite(output, combined, format='HDR-FI')
elif output.endswith(".png"):
    combined = np.clip(combined, 0, 1)
    combined = combined ** (1 / 2.2)
    combined *= 255
    imageio.imwrite(output, np.rint(combined).astype(np.uint8))
else:
    print("output format unknown.")
    exit(1)
