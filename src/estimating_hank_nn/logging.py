"""Timing, logging, and resource tracking utilities for training."""

import builtins
import os
import platform
import resource
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import torch


# %% Logging
def setup_logging(log_path):
    """Set up logging that mirrors print output to a file.

    Returns a print function that writes to both stdout and the log file.
    """
    log_file = open(log_path, "w")

    def print_fn(*args, **kwargs):
        builtins.print(*args, **kwargs)
        kwargs.pop("file", None)
        builtins.print(*args, file=log_file, **kwargs)
        log_file.flush()

    return print_fn


def log_machine_info(print_fn):
    """Log machine and software info for reproducibility tracking."""
    print_fn(f"[machine] host: {platform.node()}")
    print_fn(f"[machine] os: {platform.platform()}")
    print_fn(f"[machine] python: {platform.python_version()}, torch: {torch.__version__}, numpy: {np.__version__}")
    print_fn(f"[machine] cpu: {_cpu_model()} ({os.cpu_count()} threads)")

    ram_gb = _ram_total_gb()
    if ram_gb is not None:
        print_fn(f"[machine] ram: {ram_gb:.1f} GB")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print_fn(f"[machine] gpu: {props.name} ({props.total_memory / 1024**3:.1f} GB), "
                 f"cuda: {torch.version.cuda}, cudnn: {torch.backends.cudnn.version()}, "
                 f"driver: {_gpu_driver_version()}")
    else:
        print_fn("[machine] gpu: none")


def _gpu_driver_version():
    """NVIDIA driver version via nvidia-smi (e.g. '580.159.03'); 'unknown' if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip().splitlines()[0].strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        return "unknown"


def _cpu_model():
    """Read CPU model from /proc/cpuinfo on Linux; fall back to platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def _ram_total_gb():
    """Read total RAM in GB from /proc/meminfo on Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
    except FileNotFoundError:
        return None
    return None


# %% Timing and resource tracking
@contextmanager
def timed(label, device="cpu", print_fn=print):
    """Context manager that logs wall time and peak memory usage."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    yield
    elapsed = time.time() - t
    print_fn(f"[~] {label}: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print_fn(f"[~]   Peak CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    if device == "cuda":
        print_fn(f"[~]   Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
