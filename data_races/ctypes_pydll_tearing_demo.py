#!/usr/bin/env python3
"""
ctypes PyDLL Tearing Demo: demonstrating buffer tearing in shared memory.

This script uses `ctypes.PyDLL` and `libc.memcpy` to copy large buffers without
releasing the GIL (in standard Python). In free-threaded Python, concurrent
`memcpy` calls to the same memory region can result in "tearing," where a reader
sees a mix of data from different writers.

Run examples:
  # Free-threaded build (shows tearing)
  uv run --python 3.14t data_races/ctypes_pydll_tearing_demo.py

  # Regular build (no tearing due to GIL)
  uv run --python 3.14+gil data_races/ctypes_pydll_tearing_demo.py

  # Regular build, but using CDLL (usually shows tearing because CDLL releases the GIL)
  USE_CDLL=1 uv run --python 3.14+gil data_races/ctypes_pydll_tearing_demo.py
"""

import ctypes
import ctypes.util
import dis
import os
import threading
import platform
import sys

SIZE = 128 * 1024  # 128 KiB; bump higher if you want more chaos
ITERS = 200_000

libc_path = ctypes.util.find_library("c")
if not libc_path:
    raise RuntimeError("Couldn't find libc (this demo is for Linux/macOS).")

use_cdll = os.environ.get("USE_CDLL") == "1"
if use_cdll:
    libc = ctypes.CDLL(libc_path)
else:
    # PyDLL: like CDLL, but it does NOT release the GIL during the call (classic CPython).
    libc = ctypes.PyDLL(libc_path)

memcpy = libc.memcpy
memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
memcpy.restype = ctypes.c_void_p

memcmp = libc.memcmp
memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
memcmp.restype = ctypes.c_int

# Shared mutable buffer + reader snapshot buffer
shared = (ctypes.c_char * SIZE)()
snap = (ctypes.c_char * SIZE)()

patA = ctypes.create_string_buffer(b"A" * SIZE)
patB = ctypes.create_string_buffer(b"B" * SIZE)


def writer(src, start_barrier: threading.Barrier):
    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        return
    for _ in range(ITERS):
        memcpy(shared, src, SIZE)


def reader(start_barrier: threading.Barrier):
    try:
        start_barrier.wait()
    except threading.BrokenBarrierError:
        return
    tearing = 0
    for _ in range(ITERS):
        memcpy(snap, shared, SIZE)  # snapshot
        # Check whether snapshot is exactly A or exactly B
        if memcmp(snap, patA, SIZE) != 0 and memcmp(snap, patB, SIZE) != 0:
            tearing += 1
    print(f"tearing={tearing}")


def main():
    print("=== environment ===")
    print(f"Python ({sys.implementation.name}): {sys.version}")
    print(f"OS: {platform.platform()}, arch {platform.machine()}")
    print("===================")
    print(f"libc: {libc_path} via {type(libc).__name__}")

    start_barrier = threading.Barrier(3)

    t1 = threading.Thread(target=writer, args=(patA, start_barrier))
    t2 = threading.Thread(target=writer, args=(patB, start_barrier))
    tr = threading.Thread(target=reader, args=(start_barrier,))

    t1.start()
    t2.start()
    tr.start()
    t1.join()
    t2.join()
    tr.join()

    if os.environ.get("PRINT_BYTECODE") == "1":
        print("=== Python's bytecode of `writer` ===")
        dis.dis(writer)

        print("=== Python's bytecode of `reader` ===")
        dis.dis(reader)


if __name__ == "__main__":
    main()
