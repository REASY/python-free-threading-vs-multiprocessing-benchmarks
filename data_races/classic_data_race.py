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
"""

import os
import threading
import sys
import platform
import dis

ITERS = 1000000
THREAD_SWITCHING_INTERVAL = 0.0001


def tiny_cpu_work() -> int:
    x = 1
    for i in range(10):
        x += i % 256
    return x


class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

    def __str__(self):
        return f"({self.x}, {self.y})"


# Shared mutable object
point = Point()


def mover(has_completed_event: threading.Event):
    # This modifies two attributes - NOT atomic!
    for _ in range(ITERS):
        # Without synchronization, another thread could read between these assignments
        point.x = 1
        tiny_cpu_work()

        point.y = 2
        tiny_cpu_work()

        point.x = 3
        tiny_cpu_work()

        point.y = 4
        tiny_cpu_work()
    has_completed_event.set()


def checker(has_completed_event: threading.Event):
    inconsistent_states = 0
    while not has_completed_event.is_set():
        # Read both values
        x = point.x
        y = point.y

        # These are the only valid combinations: (0,0), (1,2), (3,4)
        # But without synchronization, we could see: (1,0), (3,2), (1,4)
        if x == 1 and y != 2:
            inconsistent_states += 1
        elif x == 3 and y != 4:
            inconsistent_states += 1
        elif x not in (0, 1, 3):
            inconsistent_states += 1

    print(f"Inconsistent states observed: {inconsistent_states}")


def main():
    print("=== environment ===")
    print(f"Python ({sys.implementation.name}): {sys.version}")
    print(f"OS: {platform.platform()}, arch {platform.machine()}")
    print(
        f"Current thread switch interval: {sys.getswitchinterval()}, setting it to {THREAD_SWITCHING_INTERVAL}"
    )
    sys.setswitchinterval(THREAD_SWITCHING_INTERVAL)
    print("===================")

    has_completed_event = threading.Event()

    t1 = threading.Thread(target=mover, args=(has_completed_event,))
    t2 = threading.Thread(target=checker, args=(has_completed_event,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print()

    if os.environ.get("PRINT_BYTECODE") == "1":
        print("=== Python's bytecode of `mover` ===")
        dis.dis(mover)


if __name__ == "__main__":
    main()
