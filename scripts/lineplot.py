import argparse
import json
import multiprocessing
import os
import re
from pathlib import Path
from typing import Dict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Groups: name, iteration, counter, run-iteration
IMAGE_OUTPUT_PATTERN = r"(.+)_(\d+)_(\d+)_(\d+)"

def imread(path):
    return imageio.v2.imread(path, format="HDR-FI")


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def mae(a, b):
    return np.mean(np.abs(a - b))


def load_run(multiprocessing_args):
    args, df_proto, run_settings, reference, run, settings = multiprocessing_args

    df = df_proto.copy()
    for iteration_dir in (args.experiment / "runs" / run).iterdir():
        if iteration_dir.is_file():
            continue
        iteration = int(iteration_dir.name)
        for image_file in iteration_dir.iterdir():
            if image_file.suffix != ".hdr":
                continue
            image = imread(image_file)
            match = re.match(IMAGE_OUTPUT_PATTERN, image_file.stem)
            assert(match)
            frame = int(match.group(2))
            df.loc[len(df.index)] = [settings[s] for s in run_settings] + [iteration, frame, int(match.group(4)), frame * args.spp, rmse(reference, image), mae(reference, image)]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="path to experiment to plot", type=Path)
    parser.add_argument("--reference", help="custom path to reference. Relative to experiment-path", type=Path, default="./reference.hdr", required=False)
    parser.add_argument("--spp", help="specify the number of samples per graph run", default=4)
    parser.add_argument("-x", default="samples")
    parser.add_argument("-y", default="rmse")
    parser.add_argument("--err-style", default="bars", choices=["bars", "band"])
    parser.add_argument("--filter", help="a python expression to use to filter the dataframe 'df'. Example: \"df['rebuild on record'] == True\"", required=False)
    parser.add_argument("--split", help="Defines splits for data (inverse of group). Defaults to all run settings", required=False, action="append")
    parser.add_argument("--xscale", default="log", choices=["linear", "log"])
    parser.add_argument("--yscale", default="log", choices=["linear", "log"])

    args = parser.parse_args()

    reference_path = args.reference if os.path.isabs(args.reference) else args.experiment / args.reference
    reference = imread(reference_path)

    with open(args.experiment / "run_info.json") as f:
        run_info: Dict = json.load(f)
    run_settings = list(next(iter(run_info.values())).keys())

    df = pd.DataFrame(columns=run_settings + ["iteration", "frame", "graph run", "samples", "rmse", "mae"])
    with multiprocessing.Pool() as pool:
        dfs = pool.map(load_run, [(args, df, run_settings, reference, run, settings) for run, settings in run_info.items()])
        df = pd.concat(dfs)
    if (args.filter):
        df = df[eval(args.filter, {"df": df})]

    split = args.split or run_settings
    sns.lineplot(df, x=args.x, y=args.y, hue=df[split].apply(tuple, axis=1), err_style=args.err_style)
    plt.xscale(args.xscale)
    plt.yscale(args.yscale)
    plt.legend(title=f"Config {tuple(split)}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
