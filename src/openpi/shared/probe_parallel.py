"""Data-parallel placement and bounded host prefetch for frozen probes.

No accelerator is initialized at import time. The training program keeps a
global batch view so masked-loss sums/counts are reduced across all devices.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging

import numpy as np


class ProbeParallel:
    def __init__(self, num_devices, batch_size):
        import jax

        if num_devices <= 0 or batch_size <= 0 or batch_size % num_devices:
            raise ValueError("Positive global batch size must be divisible by --num-devices.")
        devices = jax.local_devices()
        if len(devices) < num_devices:
            raise ValueError(
                f"Requested {num_devices} devices, but only {len(devices)} are visible. "
                "Set CUDA_VISIBLE_DEVICES before starting Python."
            )
        self.devices = devices[:num_devices]
        self.num_devices = num_devices
        self.batch_size = batch_size
        mesh = jax.sharding.Mesh(np.asarray(self.devices), ("probe_data",))
        self.replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        self.data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("probe_data"))
        logging.info(
            "Probe data parallel: devices=%s global_batch=%d per_device_batch=%d",
            self.devices, batch_size, batch_size // num_devices,
        )

    def replicate(self, tree):
        import jax

        return jax.device_put(tree, self.replicated)

    def shard(self, tree):
        import jax

        return jax.device_put(tree, self.data)


def pad_array(value, size):
    """Repeat the last row only for computation; caller masks/trims padding."""
    value = np.asarray(value)
    if value.ndim == 0 or not 0 < len(value) <= size:
        raise ValueError(f"Expected nonempty batched array with at most {size} rows, got {value.shape}.")
    if len(value) == size:
        return value
    return np.concatenate((value, np.repeat(value[-1:], size - len(value), axis=0)), axis=0)


def pad_probe_batch(batch, size):
    """Keep one compiled training/validation shape; padded rows have zero weight."""
    features, contact, force, mask = batch
    count = len(features)
    padded = [pad_array(value, size) for value in (features, contact, force, mask)]
    padded[-1] = padded[-1].copy()
    padded[-1][count:] = False
    return tuple(padded)


@contextmanager
def prefetch_batches(iterable):
    """Read exactly one next batch on a CPU thread while the current batch runs.

    No multiprocessing fork after CUDA initialization, no GPU work in this
    helper, and bounded memory. Exceptions propagate to the consumer. On early
    exit we join the reader before its dataset/video handles can be closed.
    """
    iterator = iter(iterable)
    end = object()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="probe-prefetch")

    def generate():
        future = executor.submit(next, iterator, end)
        while True:
            item = future.result()
            if item is end:
                return
            future = executor.submit(next, iterator, end)
            yield item

    batches = generate()
    try:
        yield batches
    finally:
        batches.close()
        executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def read_probe_cache(path, source, *, metrics=False):
    """Avoid loading unused multi-offset history/FAE arrays from each NPZ shard."""
    if source not in ("fae", "history"):
        raise ValueError(f"Unknown probe source: {source}")
    keys = ["frame_index", "offsets", "target_contact", "target_force", "valid_mask"]
    keys += ["fixed_hidden", "refined_hidden"] if source == "fae" else ["history_tokens"]
    if metrics:
        keys += ["episode_id", "task_id", "current_contact", "current_force"]
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in keys}
