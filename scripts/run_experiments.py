"""
Runs merian-quake to generate the results that are used for evaluation.

Config files can contain arrays of values to run every combination.
To mark arrays as combinations use two nested arrays: key: [[value1, value2, ...]]
"""

import argparse
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import imageio
import numpy as np
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

# Groups: name, iteration, counter, run-iteration
IMAGE_OUTPUT_PATTERN = r"(.+)_(\d+)_(\d+)_(\d+)"


def imread(path):
    return imageio.v2.imread(path, format="HDR-FI")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Path to the merian-quake project")
    parser.add_argument(
        "--image-path",
        help="Specify where merian-quake writes the output images (.hdr) according to the configuration",
        required=True,
    )
    parser.add_argument(
        "--reference-path",
        help="Specify the path to a reference image (.hdr) to copy to the right place",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        help="Output path for results",
        required=True,
    )
    parser.add_argument(
        "--config",
        help="Path to merian-quake JSON config. Can contain arrays of values to run multiple configurations",
        required=True,
    )

    parser.add_argument(
        "--no-compile",
        help="Disables compilation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations a configuration is run (to generate variance estimates)",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--stop-criterion",
        help="criterion to stop experiments",
        choices=["images", "iterations"],
        default="iterations",
    )
    parser.add_argument(
        "--stop", help="when to stop the experiments", type=int, required=True
    )
    parser.add_argument(
        "--gui",
        help="do not use the headless version of merian",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    return parser.parse_args()


def get_untracked(repo_path: Path) -> List[str]:
    return (
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_path
        )
        .decode()
        .splitlines()
    )


def get_untracked_patch(repo_path):
    untracked = get_untracked(repo_path)
    patches = []
    for file in untracked:
        p = subprocess.run(
            ["git", "--no-pager", "diff", "/dev/null", file],
            cwd=repo_path,
            capture_output=True,
        )
        assert p.returncode == 1
        patches.append(p.stdout.decode())
    return "\n".join(patches)


def store_working_tree(repo_path: Path, output_path: Path):
    with open(output_path / "gitinfo.txt", "w") as file:
        file.write(
            f"""\
Commit:
{subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path).decode()}

Status:
{subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_path).decode()}

Diff:
{subprocess.check_output(["git", "diff", "HEAD"], cwd=repo_path).decode()}

Untracked:
{get_untracked_patch(repo_path)}

"""
        )


def generate_configs(config_path: str, template_config_name: str):
    with open(config_path) as f:
        templated_config = json.load(f)

    # find array lengths
    lists = []

    def find_lengths(d):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], list):
                # key: [[v1, v2, ..]]
                lists.append(list(range(len(v[0]))))
            if isinstance(v, dict):
                find_lengths(v)

    find_lengths(templated_config)

    if not lists:
        # Single configuration
        yield template_config_name, templated_config
        return

    prod = itertools.product(*lists)

    def make_config(template, indices):
        d = {}
        name = []
        for k, v in template.items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                i, *indices = indices
                d[k] = v[0][i]
                name.append(f"{k}-{d[k]}")
            elif isinstance(v, dict):
                subname, d[k] = make_config(v, indices)
                name.extend(subname)
            else:
                d[k] = v
        return name, d

    for p in prod:
        name, config = make_config(templated_config, p)
        yield "_".join(name), config


def main():
    args = get_args()
    template_config_name = Path(f"exp_{args.config}").stem
    output_path = Path(args.output_path) / template_config_name

    if (args.gui):
        merian = "./bin/merian-quake"
    else:
        merian = "./bin/merian-quake-headless"

    if output_path.exists():
        logging.warning(f"output path {output_path} exists.")
        ans = input(f"Output path {output_path} exists. Continue anyway? y/[n]: ")
        if ans.lower() == "y" or ans.lower() == "yes":
            shutil.rmtree(output_path)
        else:
            exit(1)

    # Make sure experiments can be reproduced
    logging.info(f"prepare output path {output_path}")
    os.makedirs(output_path)
    logging.info(f"- copy experiment config {os.path.abspath(args.config)}")
    shutil.copy(args.config, output_path / "experiment_config.json")
    if args.reference_path:
        logging.info(f"- copy reference {os.path.abspath(args.reference_path)}")
        shutil.copy(args.reference_path, output_path / "reference.hdr")
    else:
        logging.info("- skipping copy reference (no path given)")
    logging.info("- store git info")
    store_working_tree(Path(args.project), output_path)

    with tempfile.TemporaryDirectory() as installdir:
        if (s := Path(args.image_path)).is_absolute():
            image_dir = s
        else:
            image_dir = Path(installdir) / s
        logging.info(f"expecting image output at {image_dir}")

        with tempfile.TemporaryDirectory() as builddir:
            logging.info("prepare project")
            logging.info(f"- setup builddir {builddir}")
            with open(output_path / "setuplog.txt", "w") as f:
                subprocess.run(
                    ["meson", "setup", builddir, "--prefix", installdir],
                    cwd=args.project,
                    stdout=f,
                    stderr=f,
                )
            logging.info("- compile")
            with open(output_path / "compilelog.txt", "w") as f:
                subprocess.run(["meson", "compile"], cwd=builddir, stdout=f, stderr=f)
            logging.info(f"- install to {installdir}")
            with open(output_path / "installlog.txt", "w") as f:
                subprocess.run(["meson", "install"], cwd=builddir, stdout=f, stderr=f)

        results_path = output_path / "results"
        for name, config in tqdm(
            list(generate_configs(args.config, template_config_name)), desc="config"
        ):
            run_path = results_path / name
            os.makedirs(run_path)
            run_config = run_path / "merian-quake.json"
            with open(run_config, "w") as f:
                json.dump(config, f, indent=4)

            for iteration in trange(args.iterations, desc="iteration"):
                logging.info(f"--- iteration {iteration+1:02d} ---")
                iter_path = run_path / f"{iteration:02d}"
                os.makedirs(iter_path)
                config_path = Path(installdir) / "merian-quake.json"
                config_path.unlink(True)
                logging.info(f"copy config from {run_config} to {config_path}")
                shutil.copy(run_config, config_path)

                if image_dir.exists():
                    logging.info("delete old output images")
                    for file in image_dir.iterdir():
                        if file.suffix == ".hdr":
                            file.unlink()

                logging.info(f"run experiment {name}")
                with open(iter_path / "merian-quake-log.txt", "w") as f:
                    p = subprocess.Popen(
                        [merian], cwd=installdir, stdout=f, stderr=f
                    )

                    images_found = set()
                    max_iteration = 0
                    while True:
                        r = p.poll()

                        # Check for new images
                        images = set(
                            (i for i in image_dir.iterdir() if i.suffix == ".hdr")
                            if image_dir.exists()
                            else []
                        )
                        new_images = images - images_found

                        for i, file in enumerate(sorted(new_images)):
                            match = re.match(IMAGE_OUTPUT_PATTERN, file.stem)
                            if not match:
                                logging.warning(
                                    f"image output {file.stem} does not match expected pattern"
                                )
                                continue
                            iteration = int(match.group(2))
                            max_iteration = max(max_iteration, iteration)

                            # Make sure we only check and copy images that meet the stop criterion
                            # since this process is asynchronous.
                            if (
                                args.stop_criterion == "iterations"
                                and iteration > args.stop
                            ):
                                continue
                            if (
                                args.stop_criterion == "images"
                                and len(images_found) + i >= args.stop
                            ):
                                break

                            logging.info(f"- check {file}")
                            image = imread(file.absolute())
                            if np.any(np.isnan(image)):
                                logging.warning(f"NaN values found in {file}")
                            if np.any(np.isinf(image)):
                                logging.warning(f"Inf values found in {file}")
                            logging.info(f"- copy {file}")
                            shutil.copy(file, iter_path)

                        images_found = images_found.union(new_images)

                        all_images_are_there = False
                        all_images_are_there |= (
                            args.stop_criterion == "images"
                            and len(images_found) >= args.stop
                        )
                        all_images_are_there |= (
                            args.stop_criterion == "iterations"
                            and max_iteration >= args.stop
                        )

                        if all_images_are_there:
                            p.terminate()
                            r = p.wait()
                            if r != 0:
                                logging.warning(
                                    f"merian quit with non-zero exitcode {r}"
                                )
                            break

                        if r is not None and not all_images_are_there:
                            logging.error(
                                f"merian-quake has quit prematurely with exitcode {r}."
                            )
                            break

                        time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()
