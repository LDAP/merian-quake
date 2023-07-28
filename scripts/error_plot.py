from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

# Reads a reference image from reference.hdr or uses
# the mean of all images found in a folder "reference"
#
# Compares techniques found as subfolders in "techniques"

def imread(path):
    return imageio.v2.imread(path, format="HDR-FI")

if (s := Path("reference")).exists and s.is_dir():
    ref = np.array([imread(r) for r in s.iterdir()]).mean(axis=0)
else:
    ref = imread("reference.hdr")


def rmse(img):
    return np.sqrt(np.mean((img - ref) ** 2))

def mae(img):
    return np.mean(np.abs(img - ref))

for technique in sorted(Path("techniques").iterdir()):
    errors = []
    for imgpath in sorted(technique.iterdir()):
        img = imread(imgpath)
        errors.append(rmse(img))
    plt.plot(errors, label=technique.name)

plt.xlabel("frames")
plt.ylabel("RMSE")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
