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
"""

import argparse
import gc
import os
import platform
import statistics as stats
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Protocol, Tuple


# ----------------------------
# Domain Models & Interfaces
# ----------------------------


@dataclass
class MemStats:
    rss_min: Optional[int] = None
    rss_max: Optional[int] = None
    rss_avg: Optional[float] = None
    uss_min: Optional[int] = None
    uss_max: Optional[int] = None
    uss_avg: Optional[float] = None
    pss_min: Optional[int] = None
    pss_max: Optional[int] = None
    pss_avg: Optional[float] = None


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


class Worker(Protocol):
    """Abstraction for a running unit of work (Thread or Process)."""

    def start(self) -> None: ...
    def join(self) -> None: ...
    def is_alive(self) -> bool: ...


class Backend(ABC):
    """Strategy for creating workers (Threads vs Processes)."""

    @abstractmethod
    def create_worker(self, target: Callable, args: Tuple) -> Worker:
        pass

    @abstractmethod
    def get_pids(self, workers: List[Worker]) -> List[int]:
        pass

    @abstractmethod
    def check_errors(self, workers: List[Worker]) -> None:
        """Check if any worker failed and raise RuntimeError if so."""
        pass

    def get_context_name(self) -> str:
        return "default"


class WorkloadStrategy(ABC):
    """Strategy for what the workers actually do."""

    @abstractmethod
    def get_target(self, cfg: BenchConfig) -> Callable:
        pass

    @abstractmethod
    def get_args(self, cfg: BenchConfig, worker_index: int) -> Tuple:
        pass

    @abstractmethod
    def prepare_iteration(self) -> None:
        pass

    @abstractmethod
    def start_iteration(self) -> None:
        pass


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
    work_iters: int, hold_s: float, start_evt, ready_conn
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

    def prepare_iteration(self) -> None:
        pass

    def start_iteration(self) -> None:
        pass


class MemoryWorkload(WorkloadStrategy):
    def __init__(self, ctx):
        self.ctx = ctx
        self.start_evt = None
        self.ready_conns = []

    def get_target(self, cfg: BenchConfig) -> Callable:
        return _process_entry_memory

    def get_args(self, cfg: BenchConfig, worker_index: int) -> Tuple:
        parent_conn, child_conn = self.ctx.Pipe(duplex=False)
        self.ready_conns.append(parent_conn)
        return (cfg.work_iters, cfg.hold_ms / 1000.0, self.start_evt, child_conn)

    def prepare_iteration(self) -> None:
        self.start_evt = self.ctx.Event()
        self.ready_conns = []

    def start_iteration(self) -> None:
        # Wait for all workers to be ready
        for c in self.ready_conns:
            c.recv()
            c.close()
        self.start_evt.set()


# ----------------------------
# Backend Implementation
# ----------------------------


class ThreadBackend(Backend):
    def __init__(self):
        import threading

        self.threading = threading

    def create_worker(self, target: Callable, args: Tuple) -> Worker:
        return self.threading.Thread(target=target, args=args)

    def get_pids(self, workers: List[Worker]) -> List[int]:
        return [os.getpid()]

    def check_errors(self, workers: List[Worker]) -> None:
        # Threads in Python don't easily expose exit codes like processes.
        # If we really wanted to, we could wrap the target to catch exceptions.
        # For this microbenchmark, we assume threads don't crash or we'd see it in stderr.
        pass

    # Threading needs its own implementation of MemoryWorkload helpers if we want to be strict,
    # but for now we reuse the multiprocessing ones because they work with threading.Event too.
    def Pipe(self, duplex=False):
        # Threads can just use a dummy pipe or share memory, but for simplicity
        # we can use a queue or just mock it.
        class DummyPipe:
            def send(self, val):
                pass

            def recv(self):
                return None

            def close(self):
                pass

        return DummyPipe(), DummyPipe()

    def Event(self):
        return self.threading.Event()


class ProcessBackend(Backend):
    def __init__(self, start_method: Optional[str]):
        import multiprocessing as mp

        mp.freeze_support()
        self.ctx = mp.get_context(start_method) if start_method else mp.get_context()

    def create_worker(self, target: Callable, args: Tuple) -> Worker:
        return self.ctx.Process(target=target, args=args)

    def get_pids(self, workers: List[Worker]) -> List[int]:
        # Process objects have a .pid attribute
        child_pids = [
            getattr(w, "pid") for w in workers if getattr(w, "pid") is not None
        ]
        return [os.getpid()] + child_pids

    def check_errors(self, workers: List[Worker]) -> None:
        for w in workers:
            exitcode = getattr(w, "exitcode", None)
            if exitcode is not None and exitcode != 0:
                raise RuntimeError(f"Process worker exited with {exitcode}")

    def get_context_name(self) -> str:
        return self.ctx.get_start_method()

    def Pipe(self, duplex=False):
        return self.ctx.Pipe(duplex)

    def Event(self):
        return self.ctx.Event()


# ----------------------------
# Memory Monitoring
# ----------------------------


@dataclass
class _MetricAgg:
    min_v: Optional[int] = None
    max_v: Optional[int] = None
    sum_v: int = 0
    n: int = 0

    def add(self, v: int) -> None:
        if self.min_v is None or v < self.min_v:
            self.min_v = v
        if self.max_v is None or v > self.max_v:
            self.max_v = v
        self.sum_v += v
        self.n += 1

    def avg(self) -> Optional[float]:
        return self.sum_v / self.n if self.n > 0 else None


class MemoryMonitor:
    def __init__(self, interval_s: float):
        self.interval_s = max(interval_s, 0.001)

    def sample_until(
        self, pids: Iterable[int], is_done: Callable[[], bool]
    ) -> MemStats:
        import psutil

        rss_agg, uss_agg, pss_agg = _MetricAgg(), _MetricAgg(), _MetricAgg()

        while True:
            rss_sum, uss_sum, pss_sum = 0, 0, 0
            have_uss, have_pss, saw_any = True, True, False

            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    info = p.memory_full_info()
                    rss_sum += int(info.rss)
                    saw_any = True
                    if hasattr(info, "uss"):
                        uss_sum += int(info.uss)
                    else:
                        have_uss = False
                    if hasattr(info, "pss"):
                        pss_sum += int(info.pss)
                    else:
                        have_pss = False
                except (
                    psutil.NoSuchProcess,
                    psutil.ZombieProcess,
                ):
                    continue

            if saw_any:
                rss_agg.add(rss_sum)
                if have_uss:
                    uss_agg.add(uss_sum)
                if have_pss:
                    pss_agg.add(pss_sum)

            if is_done():
                break
            time.sleep(self.interval_s)

        return MemStats(
            rss_min=rss_agg.min_v,
            rss_max=rss_agg.max_v,
            rss_avg=rss_agg.avg(),
            uss_min=uss_agg.min_v,
            uss_max=uss_agg.max_v,
            uss_avg=uss_agg.avg(),
            pss_min=pss_agg.min_v,
            pss_max=pss_agg.max_v,
            pss_avg=pss_agg.avg(),
        )


# ----------------------------
# Orchestrator
# ----------------------------


class BenchmarkRunner:
    def __init__(self, backend: Backend, workload: WorkloadStrategy):
        self.backend = backend
        self.workload = workload

    def run(self, cfg: BenchConfig) -> BenchResult:
        # Warmup
        for _ in range(cfg.warmup):
            self._run_iteration(cfg, sample_mem=False)

        per_iter: List[float] = []
        last_mem: Optional[MemStats] = None

        t0 = time.perf_counter()
        for _ in range(cfg.iterations):
            it_duration, mem = self._run_iteration(cfg, sample_mem=cfg.mem_sample)
            per_iter.append(it_duration)
            if mem:
                last_mem = mem
        wall = time.perf_counter() - t0

        return BenchResult(wall_seconds=wall, iter_seconds=per_iter, mem=last_mem)

    def _run_iteration(
        self, cfg: BenchConfig, sample_mem: bool
    ) -> Tuple[float, Optional[MemStats]]:
        gc.collect()
        self.workload.prepare_iteration()

        workers = [
            self.backend.create_worker(
                self.workload.get_target(cfg), self.workload.get_args(cfg, i)
            )
            for i in range(cfg.workers)
        ]

        it0 = time.perf_counter()
        for w in workers:
            w.start()

        self.workload.start_iteration()

        mem_stats = None
        if sample_mem:
            monitor = MemoryMonitor(cfg.mem_interval_ms / 1000.0)
            mem_stats = monitor.sample_until(
                pids=self.backend.get_pids(workers),
                is_done=lambda: all(not w.is_alive() for w in workers),
            )

        for w in workers:
            w.join()

        self.backend.check_errors(workers)

        return time.perf_counter() - it0, mem_stats


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
    res = runner.run(cfg)
    _summarize(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
