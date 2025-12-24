#!/usr/bin/env python3
"""
Scenario 2: Shared global unique set with a lock (threads vs processes)

Workload:
- Workers generate IDs, then do: (check if id in shared_set) + (insert if absent)
- The shared set is GLOBAL and the check+insert is protected by ONE lock.

Threads mode:
- shared_set: built-in set() in the same process
- lock: threading.Lock

Processes mode:
- shared_set: multiprocessing.Manager().dict() (keys are IDs)
- lock: multiprocessing context Lock shared by all workers
- This models the "default multiprocessing way": shared state = IPC/proxy + lock.

Outputs:
- ops/sec (attempted check+insert)
- inserts/sec (new unique IDs)
- dup_rate (% attempts that were duplicates)
- avg_lock_wait_ns (approx contention cost)

Run examples:
  # Free-threaded build (threads)
  uv run --with psutil --python 3.14t python scenario2_shared_unique_set.py --mode threads

  # Regular build (processes)
  uv run --with psutil --python 3.14+gil scenario2_shared_unique_set.py --mode processes

"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from benchmark_engine import (
    BenchmarkRunner,
    ProcessBackend,
    ThreadBackend,
    WorkloadStrategy,
)


# ----------------------------
# Deterministic per-worker PRNG
# ----------------------------


class SplitMix64:
    """
    Tiny fast PRNG. Deterministic, cheap, good enough for benchmarking.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)


# ----------------------------
# Benchmark config/results
# ----------------------------


@dataclass(frozen=True)
class BenchConfig:
    mode: str  # threads | processes
    workers: int
    warmup: int

    duration_ms: int
    id_space: int  # IDs are mapped to [0, id_space) to control duplicates

    mp_start: Optional[str]  # spawn | forkserver | fork | None


@dataclass(frozen=True)
class WorkerStats:
    attempts: int
    inserts: int
    lock_wait_ns: int  # total wait time (approx) in ns


@dataclass(frozen=True)
class BenchResult:
    wall_s: float
    stats: WorkerStats


# ----------------------------
# Scenario implementation
# ----------------------------


def _proc_worker_shared_unique_set(
    worker_index: int,
    id_space: int,
    duration_s: float,
    start_evt,
    lock,
    shared_dict,
    ready_conn,
    stats_conn,
) -> None:
    """
    Process worker for scenario 2.
    Must be top-level (picklable) for forkserver/spawn.
    """
    # Ready handshake (also gives parent our PID)
    try:
        ready_conn.send(os.getpid())
    finally:
        ready_conn.close()

    rng = SplitMix64(
        seed=(0xC0FFEE ^ (worker_index + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    )

    start_evt.wait()
    end = time.perf_counter() + duration_s

    attempts = 0
    inserts = 0
    lock_wait_ns = 0

    while time.perf_counter() < end:
        uid = int(rng.next_u64() % id_space)

        t0 = time.perf_counter_ns()
        lock.acquire()
        t1 = time.perf_counter_ns()
        lock_wait_ns += t1 - t0

        try:
            # Critical section: check + insert
            if uid not in shared_dict:
                shared_dict[uid] = 1
                inserts += 1
        finally:
            lock.release()

        attempts += 1

    try:
        stats_conn.send((attempts, inserts, lock_wait_ns))
    finally:
        stats_conn.close()


class SharedUniqueSetThreadsWorkload(WorkloadStrategy):
    def __init__(self):
        import threading

        self.threading = threading
        self.shared_set: set[int] = set()
        self.lock = None
        self.start_evt = None
        self.barrier = None
        self.per_worker: List[WorkerStats] = []
        self.cfg: Optional[BenchConfig] = None

    def get_target(self, cfg: BenchConfig):
        return self._worker_fn

    def get_args(self, cfg: BenchConfig, worker_index: int):
        return (worker_index,)

    def prepare_iteration(self, cfg: BenchConfig, backend) -> None:
        self.cfg = cfg
        self.shared_set.clear()
        self.lock = self.threading.Lock()
        self.start_evt = self.threading.Event()

        # Ensure a fair start: all worker threads are created and parked here before we start the timed loop.
        # Without this barrier, early threads would begin work while later threads are still being spawned,
        # skewing contention and throughput measurements.
        self.barrier = self.threading.Barrier(cfg.workers + 1)

        self.per_worker = [WorkerStats(0, 0, 0) for _ in range(cfg.workers)]

    def start_iteration(self) -> None:
        self.barrier.wait()
        self.start_evt.set()

    def collect_iteration(self):
        return _sum_stats(self.per_worker)

    def _worker_fn(self, worker_index: int) -> None:
        cfg = self.cfg
        if cfg is None:
            raise RuntimeError("Workload not initialized")

        rng = SplitMix64(
            seed=(0xC0FFEE ^ (worker_index + 1) * 0x9E3779B97F4A7C15)
            & 0xFFFFFFFFFFFFFFFF
        )

        self.barrier.wait()
        self.start_evt.wait()
        end = time.perf_counter() + (cfg.duration_ms / 1000.0)

        attempts = 0
        inserts = 0
        lock_wait_ns = 0

        while time.perf_counter() < end:
            uid = int(rng.next_u64() % cfg.id_space)

            inner_t0 = time.perf_counter_ns()
            self.lock.acquire()
            try:
                t1 = time.perf_counter_ns()
                lock_wait_ns += t1 - inner_t0

                if uid not in self.shared_set:
                    self.shared_set.add(uid)
                    inserts += 1
            finally:
                self.lock.release()

            attempts += 1

        self.per_worker[worker_index] = WorkerStats(
            attempts=attempts, inserts=inserts, lock_wait_ns=lock_wait_ns
        )


class SharedUniqueSetProcessWorkload(WorkloadStrategy):
    def __init__(self, backend: ProcessBackend):
        self.backend = backend
        self.ctx = backend.ctx
        self.manager = self.ctx.Manager()
        self.shared_dict = self.manager.dict()
        self.lock = self.ctx.Lock()
        self.start_evt = None
        self.ready_conns = []
        self.stat_conns = []
        self.duration_s = 0.0

    def get_target(self, cfg: BenchConfig):
        return _proc_worker_shared_unique_set

    def get_args(self, cfg: BenchConfig, worker_index: int):
        r_parent, r_child = self.ctx.Pipe(duplex=False)
        s_parent, s_child = self.ctx.Pipe(duplex=False)
        self.ready_conns.append(r_parent)
        self.stat_conns.append(s_parent)
        return (
            worker_index,
            cfg.id_space,
            self.duration_s,
            self.start_evt,
            self.lock,
            self.shared_dict,
            r_child,
            s_child,
        )

    def prepare_iteration(self, cfg: BenchConfig, backend) -> None:
        self.shared_dict.clear()
        self.start_evt = self.ctx.Event()
        self.ready_conns = []
        self.stat_conns = []
        self.duration_s = cfg.duration_ms / 1000.0

    def start_iteration(self) -> None:
        for c in self.ready_conns:
            c.recv()
            c.close()
        self.start_evt.set()

    def collect_iteration(self):
        per_worker_stats: List[WorkerStats] = []
        for c in self.stat_conns:
            attempts, inserts, lock_wait_ns = c.recv()
            c.close()
            per_worker_stats.append(
                WorkerStats(int(attempts), int(inserts), int(lock_wait_ns))
            )
        return _sum_stats(per_worker_stats)

    def get_extra_pids(self) -> List[int]:
        manager_pid = getattr(getattr(self.manager, "_process", None), "pid", None)
        if manager_pid is None:
            return []
        return [int(manager_pid)]

    def close(self) -> None:
        try:
            self.manager.shutdown()
        except Exception:
            pass


class Scenario2SharedUniqueSet:
    def run(self, cfg: BenchConfig) -> BenchResult:
        if cfg.mode == "threads":
            backend = ThreadBackend()
            workload: WorkloadStrategy = SharedUniqueSetThreadsWorkload()
        elif cfg.mode == "processes":
            backend = ProcessBackend(cfg.mp_start)
            workload = SharedUniqueSetProcessWorkload(backend)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

        runner = BenchmarkRunner(backend, workload)
        try:
            run_res = runner.run(
                cfg,
                warmup=cfg.warmup,
                iterations=1,
                sample_mem=False,
                mem_interval_s=0.05,
            )
        finally:
            if isinstance(workload, SharedUniqueSetProcessWorkload):
                workload.close()

        stats = run_res.payloads[0]
        if stats is None:
            raise RuntimeError("Missing worker stats from benchmark run")
        return BenchResult(wall_s=run_res.wall_seconds, stats=stats)


def _sum_stats(items: List[WorkerStats]) -> WorkerStats:
    a = sum(x.attempts for x in items)
    i = sum(x.inserts for x in items)
    w = sum(x.lock_wait_ns for x in items)
    return WorkerStats(attempts=a, inserts=i, lock_wait_ns=w)


# ----------------------------
# Reporting
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


def _print_env(cfg: BenchConfig) -> None:
    print("=== environment ===")
    print(f"Python ({sys.implementation.name}): {sys.version}")
    print(f"OS: {platform.platform()}, arch {platform.machine()}")
    print(f"Mode: {cfg.mode}")
    if cfg.mode == "processes":
        import multiprocessing as mp

        ctx = mp.get_context(cfg.mp_start) if cfg.mp_start else mp.get_context()
        print(f"mp_start_method: {ctx.get_start_method()}")
    gil = _gil_enabled_best_effort()
    if gil is not None:
        print(f"gil_enabled: {gil}")
    print(
        f"workers: {cfg.workers}  duration_ms: {cfg.duration_ms}  id_space: {cfg.id_space}"
    )
    print("===================")


def _report(res: BenchResult, cfg: BenchConfig) -> None:
    s = res.stats
    wall = max(res.wall_s, 1e-9)

    ops_s = s.attempts / wall
    inserts_s = s.inserts / wall
    dup_rate = 0.0 if s.attempts == 0 else (1.0 - (s.inserts / s.attempts)) * 100.0
    avg_lock_wait_ns = 0.0 if s.attempts == 0 else (s.lock_wait_ns / s.attempts)

    print("\n=== result ===")
    print(f"wall_s: {res.wall_s:.6f}")
    print(f"attempts: {s.attempts}  inserts: {s.inserts}  dup_rate_pct: {dup_rate:.2f}")
    print(f"ops_per_s: {ops_s:.2f}")
    print(f"inserts_per_s: {inserts_s:.2f}")
    print(f"avg_lock_wait_ns: {avg_lock_wait_ns:.1f}")

    print("==============\n")


# ----------------------------
# CLI (profile defaults)
# ----------------------------


def _default_workers() -> int:
    return 8


def parse_args() -> BenchConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["threads", "processes"], required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--duration-ms", type=int, default=None)
    ap.add_argument("--id-space", type=int, default=None)

    ap.add_argument("--mp-start", choices=["spawn", "forkserver", "fork"], default=None)

    ns = ap.parse_args()

    # Defaults per profile
    if ns.workers is None:
        ns.workers = _default_workers()

    if ns.warmup is None:
        ns.warmup = 1
    if ns.duration_ms is None:
        ns.duration_ms = 5000
    if ns.id_space is None:
        ns.id_space = 10_000_000

    if ns.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if ns.duration_ms <= 0:
        raise SystemExit("--duration-ms must be > 0")
    if ns.id_space <= 0:
        raise SystemExit("--id-space must be > 0")

    return BenchConfig(
        mode=ns.mode,
        workers=int(ns.workers),
        warmup=int(ns.warmup),
        duration_ms=int(ns.duration_ms),
        id_space=int(ns.id_space),
        mp_start=ns.mp_start,
    )


def main() -> int:
    cfg = parse_args()

    _print_env(cfg)
    res = Scenario2SharedUniqueSet().run(cfg)
    _report(res, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
