#!/usr/bin/env python3
"""
Shared benchmarking engine primitives used by scenarios.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Tuple


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


class Worker(Protocol):
    """Abstraction for a running unit of work (Thread or Process)."""

    def start(self) -> None: ...
    def join(self) -> None: ...
    def is_alive(self) -> bool: ...


class Backend(ABC):
    """Strategy for creating workers (Threads vs Processes)."""

    @abstractmethod
    def create_worker(self, target: Callable, args: Tuple) -> Worker:
        raise NotImplementedError

    @abstractmethod
    def get_pids(self, workers: List[Worker]) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def check_errors(self, workers: List[Worker]) -> None:
        """Check if any worker failed and raise RuntimeError if so."""
        raise NotImplementedError

    def get_context_name(self) -> str:
        return "default"

    def Pipe(self, duplex: bool = False):
        raise NotImplementedError

    def Event(self):
        raise NotImplementedError


class ThreadBackend(Backend):
    def __init__(self):
        import threading

        self.threading = threading

    def create_worker(self, target: Callable, args: Tuple) -> Worker:
        return self.threading.Thread(target=target, args=args)

    def get_pids(self, workers: List[Worker]) -> List[int]:
        import os

        return [os.getpid()]

    def check_errors(self, workers: List[Worker]) -> None:
        # Threads in Python don't easily expose exit codes like processes.
        # For this microbenchmark, we assume threads don't crash or we'd see it in stderr.
        pass

    def Pipe(self, duplex: bool = False):
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
        import os

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

    def Pipe(self, duplex: bool = False):
        return self.ctx.Pipe(duplex)

    def Event(self):
        return self.ctx.Event()


class WorkloadStrategy(ABC):
    """Strategy for what the workers actually do."""

    @abstractmethod
    def get_target(self, cfg) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def get_args(self, cfg, worker_index: int) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def prepare_iteration(self, cfg, backend: Backend) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_iteration(self) -> None:
        raise NotImplementedError

    def collect_iteration(self):
        return None

    def get_extra_pids(self) -> List[int]:
        return []


@dataclass(frozen=True)
class RunResult:
    wall_seconds: float
    iter_seconds: List[float]
    mem: Optional[MemStats]
    payloads: List[Optional[object]]


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


class BenchmarkRunner:
    def __init__(self, backend: Backend, workload: WorkloadStrategy):
        self.backend = backend
        self.workload = workload

    def run(
        self,
        cfg,
        warmup: int,
        iterations: int,
        sample_mem: bool,
        mem_interval_s: float,
    ) -> RunResult:
        for _ in range(warmup):
            self._run_iteration(cfg, sample_mem=False, mem_interval_s=mem_interval_s)

        per_iter: List[float] = []
        payloads: List[Optional[object]] = []
        last_mem: Optional[MemStats] = None

        t0 = time.perf_counter()
        for _ in range(iterations):
            it_duration, mem, payload = self._run_iteration(
                cfg, sample_mem=sample_mem, mem_interval_s=mem_interval_s
            )
            per_iter.append(it_duration)
            payloads.append(payload)
            if mem:
                last_mem = mem
        wall = time.perf_counter() - t0

        return RunResult(
            wall_seconds=wall, iter_seconds=per_iter, mem=last_mem, payloads=payloads
        )

    def _run_iteration(
        self, cfg, sample_mem: bool, mem_interval_s: float
    ) -> Tuple[float, Optional[MemStats], Optional[object]]:
        self.workload.prepare_iteration(cfg, self.backend)

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
            monitor = MemoryMonitor(mem_interval_s)
            pids = self.backend.get_pids(workers) + self.workload.get_extra_pids()
            mem_stats = monitor.sample_until(
                pids=pids,
                is_done=lambda: all(not w.is_alive() for w in workers),
            )

        for w in workers:
            w.join()

        self.backend.check_errors(workers)

        payload = self.workload.collect_iteration()

        return time.perf_counter() - it0, mem_stats, payload
