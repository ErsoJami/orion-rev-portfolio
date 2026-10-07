"""Reproducible runs, synchronized timings and TensorBoard logging."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def environment() -> dict:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "huggingface-hub",
        "tensorboard",
        "nbformat",
        "ipykernel",
        "psutil",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = sorted((ROOT / "experiments").glob("*.py"))
    digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sources)).hexdigest()
    return {
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "code_sha256": digest,
        "device_measurements_are_watch_measurements": False,
    }


class Run:
    def __init__(self, name: str, config: dict | None = None, root: Path | None = None):
        stamp = datetime.now(ZoneInfo("Asia/Yekaterinburg")).strftime("%Y%m%d-%H%M%S-%f")
        self.path = (root or ROOT / "runs" / "architecture") / f"{stamp}-{name}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.writer = SummaryWriter(str(self.path / "tb"), flush_secs=5)
        self.config = config or {}
        json_write(self.path / "config.json", self.config)
        json_write(self.path / "environment.json", environment())
        self.writer.add_text("run/config", json.dumps(self.config, ensure_ascii=False, indent=2))

    def log(self, values: dict, step: int = 0, prefix: str = "", *, scalar_keys=None) -> None:
        for key, value in values.items():
            if (
                (scalar_keys is None or key in scalar_keys)
                and isinstance(value, (int, float))
                and np.isfinite(value)
            ):
                self.writer.add_scalar(prefix + key, value, step)
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {"step": step, "prefix": prefix, **values},
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                )
                + "\n"
            )
        self.writer.flush()

    def save(self, filename: str, value) -> Path:
        path = self.path / filename
        json_write(path, value)
        return path

    def close(self) -> None:
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def measure(fn, device, *, warmup: int = 2, repeats: int = 10) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for _ in range(warmup):
        fn()
    sync(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        sync(device)
        started = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
        "repeats": repeats,
        "warmup": warmup,
        "rss_mib": psutil.Process().memory_info().rss / 2**20,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if torch.device(device).type == "cuda"
            else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 2**20
            if torch.device(device).type == "cuda"
            else 0.0
        ),
    }


def load_baseline(*, device: str = "cuda", dtype: str = "bfloat16", local_only: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose device='cpu'")
    if device == "cpu":
        dtype = "float32"
    from huggingface_hub import snapshot_download

    # Use a local path: this also prevents optional PEFT adapter discovery
    # issuing a HEAD request.
    cache_root = ROOT / "checkpoints" / "hf"
    checkout_cache = ROOT.parents[2] / "checkpoints" / "hf"
    if not cache_root.exists() and checkout_cache.exists():
        cache_root = checkout_cache
    snapshot = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(cache_root),
        local_files_only=local_only,
        allow_patterns=MODEL_FILES,
    )
    kwargs = dict(local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=getattr(torch, dtype), attn_implementation="sdpa", **kwargs
        )
        .to(device)
        .eval()
    )
    sync(device)
    return (
        model,
        tokenizer,
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "load_seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "dtype": dtype,
            "device": device,
        },
    )


def start_tensorboard(port: int = 6006) -> subprocess.Popen:
    import sys

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    logdir = ROOT / "runs" / "architecture"
    logdir.mkdir(parents=True, exist_ok=True)
    service = ROOT / "runs" / "tensorboard-service"
    service.mkdir(parents=True, exist_ok=True)
    with (service / f"stdout-{port}.log").open("a", encoding="utf-8") as stdout, (
        service / f"stderr-{port}.log"
    ).open("a", encoding="utf-8") as stderr:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(logdir),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--samples_per_plugin",
                "text=1000000",
            ],
            creationflags=flags,
            stdout=stdout,
            stderr=stderr,
        )


def ensure_tensorboard(port: int = 6006, timeout: float = 20) -> dict:
    """Check the HTTP logdir, reuse the matching server, or start our Python module.

    Explicit module launch avoids Windows PATH selecting an unrelated tensorboard.exe.
    An unrelated server is preserved; the next port is used instead.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    expected = os.path.normcase(str((ROOT / "runs" / "architecture").resolve()))
    for selected in range(port, port + 5):
        url = f"http://127.0.0.1:{selected}"
        try:
            with opener.open(url + "/data/logdir", timeout=1) as response:
                current = json.load(response)["logdir"]
            if os.path.normcase(str(Path(current).resolve())) == expected:
                return {"url": url, "reused": True, "logdir": current}
            continue
        except (OSError, ValueError, KeyError):
            pass
        process = start_tensorboard(selected)
        started = time.perf_counter()
        while time.perf_counter() - started < timeout:
            if process.poll() is not None:
                raise RuntimeError(
                    f"TensorBoard exited; read runs/tensorboard-service/stderr-{selected}.log"
                )
            try:
                with opener.open(url + "/data/logdir", timeout=1) as response:
                    current = json.load(response)["logdir"]
                if os.path.normcase(str(Path(current).resolve())) == expected:
                    return {"url": url, "pid": process.pid, "reused": False, "logdir": current}
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.2)
        process.terminate()
        raise TimeoutError("TensorBoard failed to become ready; inspect service logs")
    raise RuntimeError("no free TensorBoard port in the selected range")
