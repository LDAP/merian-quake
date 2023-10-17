# Reads a reference image from reference.hdr or uses
# the mean of all images found in a folder "reference"
#
# Compares techniques found as subfolders in "techniques"

import sys
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

prefix = Path(".")
if len(sys.argv) > 1:
    prefix = Path(sys.argv[1])

def imread(path):
    return imageio.v2.imread(path, format="HDR-FI")

if (s := prefix / Path("reference")).exists and s.is_dir():
    ref = np.array([imread(r) for r in s.iterdir()]).mean(axis=0)
else:
    ref = imread(prefix / "reference.hdr")


def rmse(img):
    return np.sqrt(np.mean((img - ref) ** 2))

def mae(img):
    return np.mean(np.abs(img - ref))

for technique in sorted((prefix / "techniques").iterdir()):
    if technique.name.startswith("_"):
        continue

    errors = []
    frames = []
    for imgpath in sorted(technique.iterdir()):
        if imgpath.suffix != ".hdr":
            continue
        frame = int(imgpath.stem.split("_")[2])
        img = imread(imgpath)
        errors.append(rmse(img))
        frames.append(frame + 1) # frames start with 0
    plt.plot(frames, errors, label=technique.name)

plt.xlabel("frames")
plt.ylabel("RMSE")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
