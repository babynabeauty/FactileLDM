"""Import the dataset API from either supported LeRobot package layout."""

import importlib


def import_lerobot_dataset():
    try:
        return importlib.import_module("lerobot.common.datasets.lerobot_dataset")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "lerobot.common",
            "lerobot.common.datasets",
            "lerobot.common.datasets.lerobot_dataset",
        }:
            raise
    return importlib.import_module("lerobot.datasets.lerobot_dataset")
