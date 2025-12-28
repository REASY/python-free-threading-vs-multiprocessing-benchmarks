#!/usr/bin/env python3
"""
Classic Data Race: demonstrating non-atomic updates to a shared Python object.

This script shows how multiple threads can observe inconsistent states of an object
when updates to its attributes are not synchronized (e.g., (x=1, y=2) vs (x=3, y=4)).

Run examples:
  # Free-threaded build (will likely show inconsistent states)
  uv run --python 3.14t data_races/classic_data_race.py

  # Regular build
  uv run --python 3.14+gil data_races/classic_data_race.py

  # Regular build, but widen the race window to make the issue obvious
  WIDEN=1 uv run --python 3.14+gil data_races/classic_data_race.py
"""

import threading
import sys
import platform
import dis

from util import env_int, env_float, env_flag

DEFAULT_ITERS = 1_000_000
DEFAULT_SWITCH_INTERVAL = 0.0001
DEFAULT_WORK_ITERS = 10


ITERS = env_int("ITERS", DEFAULT_ITERS)
THREAD_SWITCHING_INTERVAL = env_float("SWITCH_INTERVAL", DEFAULT_SWITCH_INTERVAL)
WIDEN = env_flag("WIDEN")
WORK_ITERS = env_int("WORK_ITERS", DEFAULT_WORK_ITERS)


def tiny_cpu_work(work_iters: int) -> int:
    x = 1
    for i in range(work_iters):
        x += i & 255
    return x


class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

    def __str__(self):
        return f"({self.x}, {self.y})"


# Shared mutable object
point = Point()

VALID_STATES = {(0, 0), (1, 2), (3, 4)}


def mover(done: threading.Event, start_barrier: threading.Barrier):
    start_barrier.wait()
    work_iters = WORK_ITERS

    if not WIDEN:
        for _ in range(ITERS):
            # Without synchronization, another thread could read between these assignments
            point.x = 1
            point.y = 2
            point.x = 3
            point.y = 4
        done.set()
        return

    for _ in range(ITERS):
        # Without synchronization, another thread could read between these assignments
        point.x = 1
        tiny_cpu_work(work_iters)

        point.y = 2
        tiny_cpu_work(work_iters)

        point.x = 3
        tiny_cpu_work(work_iters)

        point.y = 4
        tiny_cpu_work(work_iters)
    done.set()


def checker(done: threading.Event, start_barrier: threading.Barrier):
    inconsistent_states = 0
    start_barrier.wait()

    while not done.is_set():
        # Read both values
        x = point.x
        y = point.y

        # These are the only valid combinations: (0,0), (1,2), (3,4)
        # But without synchronization, we could see: (1,0), (3,2), (1,4), etc.
        if (x, y) not in VALID_STATES:
            inconsistent_states += 1

    print(f"Inconsistent states observed: {inconsistent_states}")


def main():
    print("=== environment ===")
    print(f"Python ({sys.implementation.name}): {sys.version}")
    print(f"OS: {platform.platform()}, arch {platform.machine()}")
    print(f"Config: ITERS={ITERS}, WIDEN={WIDEN}, WORK_ITERS={WORK_ITERS}")
    print(
        f"Current thread switch interval: {sys.getswitchinterval()}, setting it to {THREAD_SWITCHING_INTERVAL}"
    )
    sys.setswitchinterval(THREAD_SWITCHING_INTERVAL)
    print("===================")

    has_completed_event = threading.Event()
    start_barrier = threading.Barrier(2)

    t1 = threading.Thread(target=mover, args=(has_completed_event, start_barrier))
    t2 = threading.Thread(target=checker, args=(has_completed_event, start_barrier))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print()

    if env_flag("PRINT_BYTECODE"):
        print("=== Python's bytecode of `mover` ===")
        dis.dis(mover)


if __name__ == "__main__":
    main()
