#!/usr/bin/env python3
"""
Scenario 3: Single-writer (or sharded-writer) process for unique set (processes only)

Workload:
- N producer processes generate IDs and send them to one writer process (or shard).
- Each writer owns its set and performs check+insert serially (no lock).
- Sharding uses `uid % writers` so the same ID always goes to the same writer.
- Producers block for a per-item ACK (duplicate vs insert) from the writer.
- This removes Manager-proxy + shared-lock overhead from Scenario 2, but adds IPC round trips.

Outputs:
- ops/sec (attempted check+insert)
- inserts/sec (new unique IDs)
- dup_rate (% attempts that were duplicates)
- avg queue put latency (ns)
- avg ACK wait latency (ns)

Run examples:
  uv run --python 3.14+gil python scenario3_sharded_writer.py
  uv run --python 3.14+gil python scenario3_sharded_writer.py --writers 4
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, cast

from multiprocessing import queues, synchronize
from multiprocessing.connection import Connection

from benchmark_engine import BenchmarkRunner, ProcessBackend, WorkloadStrategy
from common import SplitMix64


@dataclass(frozen=True)
class BenchConfig:
    workers: int
    warmup: int
    duration_ms: int
    id_space: int
    writers: int
    mp_start: Optional[str]  # spawn | forkserver | fork | None


@dataclass(frozen=True)
class WorkerStats:
    attempts: int
    inserts: int
    queue_put_ns: int
    ack_wait_ns: int


@dataclass(frozen=True)
class BenchResult:
    wall_s: float
    stats: WorkerStats


def _proc_producer_single_writer(
    worker_index: int,
    id_space: int,
    duration_s: float,
    writer_count: int,
    start_evt: synchronize.Event,
    writer_queues: List[queues.SimpleQueue],
    ack_queue: queues.SimpleQueue,
    ready_conn: Connection,
    stats_conn: Connection,
) -> None:
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
    queue_put_ns = 0
    ack_wait_ns = 0

    while time.perf_counter() < end:
        uid = int(rng.next_u64() % id_space)
        shard = uid % writer_count
        t0 = time.perf_counter_ns()
        writer_queues[shard].put((worker_index, uid))
        t1 = time.perf_counter_ns()
        queue_put_ns += t1 - t0

        t2 = time.perf_counter_ns()
        ack_queue.get()
        t3 = time.perf_counter_ns()
        ack_wait_ns += t3 - t2
        attempts += 1

    for q in writer_queues:
        q.put((worker_index, None))
    try:
        stats_conn.send((attempts, queue_put_ns, ack_wait_ns))
    finally:
        stats_conn.close()


def _proc_writer_single_writer(
    shard_index: int,
    worker_count: int,
    start_evt: synchronize.Event,
    queue: queues.SimpleQueue,
    ack_queues: List[queues.SimpleQueue],
    ready_conn: Connection,
    stats_conn: Connection,
) -> None:
    try:
        ready_conn.send(os.getpid())
    finally:
        ready_conn.close()

    del shard_index
    start_evt.wait()

    shared_set: set[int] = set()
    attempts = 0
    inserts = 0
    done = 0

    while done < worker_count:
        worker_index, uid = queue.get()
        if uid is None:
            done += 1
            continue
        attempts += 1
        if uid not in shared_set:
            shared_set.add(uid)
            inserts += 1
            ack_queues[worker_index].put(True)
        else:
            ack_queues[worker_index].put(False)

    try:
        stats_conn.send((attempts, inserts))
    finally:
        stats_conn.close()


class SingleWriterProcessWorkload(WorkloadStrategy):
    def __init__(self, backend: ProcessBackend):
        self.backend = backend
        self.ctx = backend.ctx
        self.writer_queues: List[queues.SimpleQueue] = []
        self.ack_queues: List[queues.SimpleQueue] = []
        self.start_evt: Optional[synchronize.Event] = None
        self.ready_conns: List[Connection] = []
        self.writer_ready: List[Connection] = []
        self.writer_stats: List[Connection] = []
        self.writer_workers: List[Any] = []
        self.producer_stats: List[Connection] = []
        self.producer_stats_child: List[Connection] = []
        self.duration_s = 0.0
        self.worker_count = 0
        self.writer_count = 0

    def get_target(self, cfg: BenchConfig):
        return _proc_producer_single_writer

    def get_args(self, cfg: BenchConfig, worker_index: int):
        r_parent, r_child = self.ctx.Pipe(duplex=False)
        self.ready_conns.append(r_parent)
        stats_conn = self.producer_stats_child[worker_index]
        return (
            worker_index,
            cfg.id_space,
            self.duration_s,
            self.writer_count,
            self.start_evt,
            self.writer_queues,
            self.ack_queues[worker_index],
            r_child,
            stats_conn,
        )

    def prepare_iteration(self, cfg: BenchConfig, backend) -> None:
        self.writer_queues = []
        self.ack_queues = []
        self.start_evt = self.ctx.Event()
        self.ready_conns = []
        self.duration_s = cfg.duration_ms / 1000.0
        self.worker_count = cfg.workers
        self.writer_count = cfg.writers
        self.writer_ready = []
        self.writer_stats = []
        self.writer_workers = []
        self.producer_stats = []
        self.producer_stats_child = []

        for _ in range(cfg.workers):
            self.ack_queues.append(self.ctx.SimpleQueue())
            stats_parent, stats_child = self.ctx.Pipe(duplex=False)
            self.producer_stats.append(stats_parent)
            self.producer_stats_child.append(stats_child)

        for shard in range(cfg.writers):
            self.writer_queues.append(self.ctx.SimpleQueue())
            wr_parent, wr_child = self.ctx.Pipe(duplex=False)
            ws_parent, ws_child = self.ctx.Pipe(duplex=False)
            self.writer_ready.append(wr_parent)
            self.writer_stats.append(ws_parent)
            worker = self.backend.create_worker(
                _proc_writer_single_writer,
                (
                    shard,
                    cfg.workers,
                    self.start_evt,
                    self.writer_queues[shard],
                    self.ack_queues,
                    wr_child,
                    ws_child,
                ),
            )
            self.writer_workers.append(worker)

    def start_iteration(self) -> None:
        for w in self.writer_workers:
            w.start()

        for c in self.ready_conns:
            c.recv()
            c.close()

        for c in self.writer_ready:
            c.recv()
            c.close()

        start_evt = self.start_evt
        assert start_evt is not None
        start_evt.set()

    def collect_iteration(self):
        attempts = 0
        inserts = 0
        for c in self.writer_stats:
            shard_attempts, shard_inserts = c.recv()
            c.close()
            attempts += int(shard_attempts)
            inserts += int(shard_inserts)

        queue_put_ns = 0
        ack_wait_ns = 0
        for c in self.producer_stats:
            _prod_attempts, prod_put_ns, prod_ack_ns = c.recv()
            c.close()
            queue_put_ns += int(prod_put_ns)
            ack_wait_ns += int(prod_ack_ns)

        for w in self.writer_workers:
            w.join()
            exitcode = getattr(w, "exitcode", None)
            if exitcode is not None and exitcode != 0:
                raise RuntimeError(f"Writer process exited with {exitcode}")

        return WorkerStats(attempts, inserts, queue_put_ns, ack_wait_ns)

    def get_extra_pids(self) -> List[int]:
        pids: List[int] = []
        for w in self.writer_workers:
            pid = getattr(w, "pid", None)
            if pid is not None:
                pids.append(int(pid))
        return pids


class Scenario3SingleWriter:
    def run(self, cfg: BenchConfig) -> BenchResult:
        backend = ProcessBackend(cfg.mp_start)
        workload = SingleWriterProcessWorkload(backend)

        runner = BenchmarkRunner(backend, workload)
        run_res = runner.run(
            cfg,
            warmup=cfg.warmup,
            iterations=1,
            sample_mem=False,
            mem_interval_s=0.05,
        )

        stats = run_res.payloads[0]
        if stats is None:
            raise RuntimeError("Missing worker stats from benchmark run")
        return BenchResult(wall_s=run_res.wall_seconds, stats=cast(WorkerStats, stats))


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
    import multiprocessing as mp

    ctx = mp.get_context(cfg.mp_start) if cfg.mp_start else mp.get_context()
    print(f"mp_start_method: {ctx.get_start_method()}")
    gil = _gil_enabled_best_effort()
    if gil is not None:
        print(f"gil_enabled: {gil}")
    print(
        "workers: "
        f"{cfg.workers}  writers: {cfg.writers}  duration_ms: {cfg.duration_ms}  "
        f"id_space: {cfg.id_space}"
    )
    print("===================")


def _report(res: BenchResult, cfg: BenchConfig) -> None:
    s = res.stats
    wall = max(res.wall_s, 1e-9)

    ops_s = s.attempts / wall
    inserts_s = s.inserts / wall
    dup_rate = 0.0 if s.attempts == 0 else (1.0 - (s.inserts / s.attempts)) * 100.0
    avg_put_ns = s.queue_put_ns / max(s.attempts, 1)
    avg_ack_ns = s.ack_wait_ns / max(s.attempts, 1)

    print("\n=== result ===")
    print(f"wall_s: {res.wall_s:.6f}")
    print(f"attempts: {s.attempts}  inserts: {s.inserts}  dup_rate_pct: {dup_rate:.2f}")
    print(f"ops_per_s: {ops_s:.2f}")
    print(f"inserts_per_s: {inserts_s:.2f}")
    print(f"avg_queue_put_ns: {avg_put_ns:.0f}")
    print(f"avg_ack_wait_ns: {avg_ack_ns:.0f}")
    print("==============\n")


def _default_workers() -> int:
    return 8


def parse_args() -> BenchConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--duration-ms", type=int, default=None)
    ap.add_argument("--id-space", type=int, default=None)
    ap.add_argument("--writers", type=int, default=None)
    ap.add_argument("--mp-start", choices=["spawn", "forkserver", "fork"], default=None)

    ns = ap.parse_args()

    if ns.workers is None:
        ns.workers = _default_workers()
    if ns.warmup is None:
        ns.warmup = 1
    if ns.duration_ms is None:
        ns.duration_ms = 5000
    if ns.id_space is None:
        ns.id_space = 10_000_000
    if ns.writers is None:
        ns.writers = 1

    if ns.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if ns.writers <= 0:
        raise SystemExit("--writers must be > 0")
    if ns.duration_ms <= 0:
        raise SystemExit("--duration-ms must be > 0")
    if ns.id_space <= 0:
        raise SystemExit("--id-space must be > 0")

    return BenchConfig(
        workers=int(ns.workers),
        warmup=int(ns.warmup),
        duration_ms=int(ns.duration_ms),
        id_space=int(ns.id_space),
        writers=int(ns.writers),
        mp_start=ns.mp_start,
    )


def main() -> int:
    cfg = parse_args()
    _print_env(cfg)
    res = Scenario3SingleWriter().run(cfg)
    _report(res, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
