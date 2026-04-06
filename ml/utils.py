import os
import re
from pathlib import Path
import csv

import torch
import fsspec


def sort_latest_checkpoints(experiment_dir: str) -> list[str]:
    def extract_number(dir_name: str) -> int:
        match = re.search(r"cp-(\d+)\.ckpt", dir_name)
        if match:
            step = int(match.group(1))
            return step

    # get dirs
    fs = fsspec.get_mapper(experiment_dir).fs
    checkpoint_dirs = fs.ls(experiment_dir, detail=False)
    # filter
    checkpoint_dirs = list(filter(extract_number, checkpoint_dirs))
    # sort
    checkpoint_dirs_sorted = sorted(checkpoint_dirs, key=extract_number)
    return checkpoint_dirs_sorted


def load_most_recent_checkpoint(experiment_dir: str) -> str | None:
    checkpoints = sort_latest_checkpoints(experiment_dir)
    if len(checkpoints) == 0:
        return None
    else:
        return checkpoints[-1]


def save_checkpoint(
    out_dir: str,
    global_step: int,
    state_dict: dict[str, any],
    should_delete_old: bool = True,
    keep: int = 100,
):
    # list old
    fs = fsspec.get_mapper(out_dir).fs
    checkpoints_old = sort_latest_checkpoints(out_dir)

    # update state dict
    state_dict["global_step"] = global_step
    # save
    out_path = f"{out_dir}/cp-{global_step:07d}.ckpt"
    with fsspec.open(out_path, "wb") as f:
        torch.save(state_dict, f)
    # delete old except last one
    if should_delete_old:
        for f in checkpoints_old[:-keep]:
            fs.rm(f)


class CSVLogger:
    def __init__(self, exp_dir: str) -> None:
        for i in range(1000):
            csv_path = Path(f"{exp_dir}/log/version_{i}/metrics.csv")
            if not csv_path.exists():
                break
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = str(csv_path)
        self.counter = 0

    def write(self, metrics: dict) -> None:
        if self.counter == 0:  # header
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(metrics.keys())

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metrics.values())
        self.counter += 1


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)



def weights_interpolation(weights: dict, i: int, max_steps: int) -> dict:
    weights_new = {}
    for k, v in weights.items():
        if isinstance(v, list):
            assert len(v) == 2
            weights_new[k] = np.interp(i, [0, max_steps], v)
        else:
            weights_new[k] = v
    return weights_new

