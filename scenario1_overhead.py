#!/usr/bin/env python3
"""
Scenario 1: thread/process creation overhead + optional memory sampling.

Profiles
- overhead: tight create/start/join microbenchmark (minimal instrumentation)
- memory: keep workers alive long enough to sample memory_full_info() (RSS/USS/PSS)

Notes
- In processes mode, memory is reported as SUM across parent + children PIDs.
  SUM(RSS) overcounts shared pages; PSS is best when available (Linux).
- Memory sampling itself costs CPU. For timing comparisons, trust the overhead profile.

Run examples:
  # Free-threaded build (threads)
  uv run --python 3.14t scenario1_overhead.py --mode threads

  # Regular build (processes)
  uv run --python 3.14+gil scenario1_overhead.py --mode processes
"""

import argparse
import gc
import os
import platform
import statistics as stats
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from benchmark_engine import (
    Backend,
    BenchmarkRunner,
    MemStats,
    ProcessBackend,
    ThreadBackend,
    WorkloadStrategy,
)

from multiprocessing.connection import Connection
from multiprocessing import synchronize

# ----------------------------
# Domain Models & Interfaces
# ----------------------------


@dataclass(frozen=True)
class BenchConfig:
    mode: str  # threads | processes
    profile: str  # overhead | memory
    iterations: int
    workers: int
    work_iters: int
    warmup: int
    mp_start: Optional[str]  # spawn | forkserver | fork | None

    # memory profile knobs
    hold_ms: int
    mem_interval_ms: int
    mem_sample: bool


@dataclass(frozen=True)
class BenchResult:
    wall_seconds: float
    iter_seconds: List[float]
    mem: Optional[MemStats]


# ----------------------------
# Workload Implementation
# ----------------------------


def _cpu_tiny_work(work_iters: int) -> int:
    """Tiny deterministic CPU workload in pure Python."""
    x = 0x12345678
    for _ in range(work_iters):
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        x ^= x >> 16
    return x


def _busy_run_for(work_iters: int, hold_s: float) -> None:
    """Run CPU work for at least hold_s seconds."""
    end = time.perf_counter() + hold_s
    while time.perf_counter() < end:
        _cpu_tiny_work(work_iters)


def _process_entry_overhead(work_iters: int) -> None:
    _cpu_tiny_work(work_iters)


def _process_entry_memory(
    work_iters: int, hold_s: float, start_evt: synchronize.Event, ready_conn: Connection
) -> None:
    # Make the process "real" before we signal ready (imports, allocator, etc.)
    _cpu_tiny_work(1000)
    try:
        if ready_conn:
            ready_conn.send(os.getpid())
    finally:
        if ready_conn:
            ready_conn.close()

    if start_evt:
        start_evt.wait()
    _busy_run_for(work_iters, hold_s)


class OverheadWorkload(WorkloadStrategy):
    def get_target(self, cfg: BenchConfig) -> Callable:
        return _process_entry_overhead

    def get_args(self, cfg: BenchConfig, worker_index: int) -> Tuple:
        return (cfg.work_iters,)

    def prepare_iteration(self, cfg: BenchConfig, backend: Backend) -> None:
        gc.collect()

    def start_iteration(self) -> None:
        pass


class MemoryWorkload(WorkloadStrategy):
    def __init__(self, backend: Backend):
        self.backend = backend
        self.start_evt: Optional[synchronize.Event] = None
        self.ready_conns: List[Connection] = []

    def get_target(self, cfg: BenchConfig) -> Callable:
        return _process_entry_memory

    def get_args(self, cfg: BenchConfig, worker_index: int) -> Tuple:
        parent_conn, child_conn = self.backend.Pipe(duplex=False)
        self.ready_conns.append(parent_conn)
        start_evt = self.start_evt
        assert start_evt is not None
        return (cfg.work_iters, cfg.hold_ms / 1000.0, start_evt, child_conn)

    def prepare_iteration(self, cfg: BenchConfig, backend: Backend) -> None:
        gc.collect()
        self.start_evt = self.backend.Event()
        self.ready_conns = []

    def start_iteration(self) -> None:
        # Wait for all workers to be ready
        for c in self.ready_conns:
            c.recv()
            c.close()
        start_evt = self.start_evt
        assert start_evt is not None
        start_evt.set()


# ----------------------------
# CLI & Main
# ----------------------------


def _gil_enabled_best_effort() -> Optional[bool]:
    for attr in ("_is_gil_enabled", "_get_gil_enabled"):
        fn = getattr(sys, attr, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    return None


def _print_env(cfg: BenchConfig, backend: Backend) -> None:
    print("=== environment ===")
    print(f"Python ({sys.implementation.name}): {sys.version}")
    print(f"OS: {platform.platform()}, arch {platform.machine()}")
    print(f"Mode: {cfg.mode}")
    print(f"Profile: {cfg.profile}")

    if cfg.mode == "processes":
        print(f"mp_start_method: {backend.get_context_name()}")

    gil = _gil_enabled_best_effort()
    if gil is not None:
        print(f"gil_enabled: {gil}")

    print(
        f"workers: {cfg.workers}  iterations: {cfg.iterations}  warmup: {cfg.warmup}  work_iters: {cfg.work_iters}"
    )

    if cfg.profile == "memory" or cfg.mem_sample:
        print(
            f"hold_ms: {cfg.hold_ms}  mem_sample: {cfg.mem_sample}  mem_interval_ms: {cfg.mem_interval_ms}"
        )

    print("===================")


def _fmt_val(x: Optional[float]) -> str:
    return "n/a" if x is None else str(int(x))


def _summarize(res: BenchResult) -> None:
    it = res.iter_seconds
    mean = stats.mean(it) if it else float("nan")
    p50 = stats.median(it) if it else float("nan")
    p95 = stats.quantiles(it, n=20)[18] if len(it) >= 20 else float("nan")

    print("\n=== result ===")
    print(f"total_wall_s: {res.wall_seconds:.6f}")
    print(f"avg_iter_s: {mean:.9f}")
    print(f"p50_iter_s: {p50:.9f}")
    print(f"p95_iter_s: {p95:.9f}")

    if res.mem:
        m = res.mem
        print("\n--- memory (bytes) via psutil.memory_full_info() ---")
        print(
            f"rss_min: {_fmt_val(m.rss_min)}  rss_max: {_fmt_val(m.rss_max)}  rss_avg: {_fmt_val(m.rss_avg)}"
        )
        print(
            f"uss_min: {_fmt_val(m.uss_min)}  uss_max: {_fmt_val(m.uss_max)}  uss_avg: {_fmt_val(m.uss_avg)}"
        )
        print(
            f"pss_min: {_fmt_val(m.pss_min)}  pss_max: {_fmt_val(m.pss_max)}  pss_avg: {_fmt_val(m.pss_avg)}"
        )
        print("---------------------------------------------------")
    else:
        print("\nmemory: (disabled)")

    print("==============\n")


def parse_args() -> BenchConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["threads", "processes"], required=True)
    ap.add_argument("--profile", choices=["overhead", "memory"], default="overhead")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--work-iters", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument(
        "--mp-start",
        choices=["spawn", "forkserver", "fork"],
        default=None,
        help="Only for --mode processes.",
    )
    ap.add_argument("--hold-ms", type=int, default=None)
    ap.add_argument("--mem-interval-ms", type=int, default=None)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mem-sample", dest="mem_sample", action="store_true")
    g.add_argument("--no-mem-sample", dest="mem_sample", action="store_false")
    ap.set_defaults(mem_sample=None)

    ns = ap.parse_args()

    # Defaults logic
    if ns.workers is None:
        ns.workers = 8
    if ns.work_iters is None:
        ns.work_iters = 20_000

    if ns.profile == "overhead":
        ns.iterations = ns.iterations or 1000
        ns.warmup = ns.warmup if ns.warmup is not None else 5
        ns.hold_ms = ns.hold_ms or 0
        ns.mem_interval_ms = ns.mem_interval_ms or 50
        ns.mem_sample = ns.mem_sample if ns.mem_sample is not None else False
    else:
        ns.iterations = ns.iterations or 30
        ns.warmup = ns.warmup if ns.warmup is not None else 2
        ns.hold_ms = ns.hold_ms or 1000
        ns.mem_interval_ms = ns.mem_interval_ms or 50
        ns.mem_sample = ns.mem_sample if ns.mem_sample is not None else True

    return BenchConfig(
        mode=ns.mode,
        profile=ns.profile,
        iterations=int(ns.iterations),
        workers=int(ns.workers),
        work_iters=int(ns.work_iters),
        warmup=int(ns.warmup),
        mp_start=ns.mp_start,
        hold_ms=int(ns.hold_ms),
        mem_interval_ms=int(ns.mem_interval_ms),
        mem_sample=bool(ns.mem_sample),
    )


def main() -> int:
    cfg = parse_args()

    if cfg.mem_sample:
        try:
            import psutil  # noqa: F401
        except ImportError:
            raise SystemExit("psutil is required for memory sampling.")

    if cfg.mode == "threads":
        backend = ThreadBackend()
    else:
        backend = ProcessBackend(cfg.mp_start)

    if cfg.profile == "overhead":
        workload = OverheadWorkload()
    else:
        workload = MemoryWorkload(backend)

    _print_env(cfg, backend)
    runner = BenchmarkRunner(backend, workload)
    run_res = runner.run(
        cfg,
        warmup=cfg.warmup,
        iterations=cfg.iterations,
        sample_mem=cfg.mem_sample,
        mem_interval_s=cfg.mem_interval_ms / 1000.0,
    )
    res = BenchResult(
        wall_seconds=run_res.wall_seconds,
        iter_seconds=run_res.iter_seconds,
        mem=run_res.mem,
    )
    _summarize(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
