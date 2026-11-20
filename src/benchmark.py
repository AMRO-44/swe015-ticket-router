"""Area D - benchmarking: model size on disk and single-message CPU latency (R4)."""
import os
import platform
import time
from pathlib import Path

import numpy as np

from .config import LATENCY_RUNS, LATENCY_WARMUP

BENCH_MESSAGES = [
    "my dishwasher is not draining the water model DW-6534",
    "what is my bank account balance",
    "bulaşık makinem su boşaltmıyor",
    "how do i book a flight to istanbul",
    "can you tell me a joke",
    "i lost my credit card please freeze it",
    "what's the weather tomorrow",
    "my oven won't turn on at all",
    "set an alarm for 7 am",
    "who won the football game last night",
]


def hardware_string() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    threads = os.cpu_count()
    try:
        import torch
        threads = torch.get_num_threads()
    except Exception:
        pass
    return f"{cpu}, {os.cpu_count()} logical cores, {threads} threads, {platform.system()} (CPU only, batch size 1)"


def size_mb(*paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return round(total / 1e6, 2)


def latency_ms(predict_one, runs=LATENCY_RUNS, warmup=LATENCY_WARMUP) -> dict:
    """predict_one(text) must run the full model on ONE message (preprocessing included)."""
    for i in range(warmup):
        predict_one(BENCH_MESSAGES[i % len(BENCH_MESSAGES)])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        predict_one(BENCH_MESSAGES[i % len(BENCH_MESSAGES)])
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"latency_ms": round(float(t.mean()), 3), "latency_p95_ms": round(float(np.percentile(t, 95)), 3),
            "latency_runs": runs}
