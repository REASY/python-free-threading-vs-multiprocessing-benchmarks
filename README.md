# Reproducible micro-benchmarks for Python 3.14 free-threading: threads vs. multiprocessing overhead

## Development

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install Python 3.14 with Free-Threading `uv python install 3.14t`

## Usage

### Scenario1

#### Threads (Free-threading)

##### Profile overhead

```console
✗ uv run --python 3.14t scenario1.py --mode threads
=== environment ===
Python (cpython): 3.14.2 free-threading build (main, Dec  9 2025, 19:03:17) [Clang 21.1.4 ]
OS: Linux-6.17.0-8-generic-x86_64-with-glibc2.42, arch x86_64
Mode: threads
Profile: overhead
gil_enabled: False
workers: 8  iterations: 1000  warmup: 5  work_iters: 20000
===================

=== result ===
total_wall_s: 2.458765
avg_iter_s: 0.001822840
p50_iter_s: 0.001771547
p95_iter_s: 0.002239553

memory: (disabled)
==============
```

##### Profile memory

```console
✗ uv run --python 3.14t scenario1.py --mode threads --profile memory 
=== environment ===
Python (cpython): 3.14.2 free-threading build (main, Dec  9 2025, 19:03:17) [Clang 21.1.4 ]
OS: Linux-6.17.0-8-generic-x86_64-with-glibc2.42, arch x86_64
Mode: threads
Profile: memory
gil_enabled: False
workers: 8  iterations: 30  warmup: 2  work_iters: 20000
hold_ms: 1000  mem_sample: True  mem_interval_ms: 50
===================

=== result ===
total_wall_s: 30.265476
avg_iter_s: 1.007158850
p50_iter_s: 1.006997269
p95_iter_s: 1.009170881

--- memory (bytes) via psutil.memory_full_info() ---
rss_min: 38490112  rss_max: 39014400  rss_avg: 38989433
uss_min: 35491840  uss_max: 36016128  uss_avg: 35991161
pss_min: 35523584  pss_max: 36047872  pss_avg: 36022905
---------------------------------------------------
==============
```

#### Processes

##### Profile overhead

```console
✗ uv run --python 3.14+gil scenario1.py --mode processes
=== environment ===
Python (cpython): 3.14.2 (main, Dec  9 2025, 19:03:28) [Clang 21.1.4 ]
OS: Linux-6.17.0-8-generic-x86_64-with-glibc2.42, arch x86_64
Mode: processes
Profile: overhead
mp_start_method: forkserver
gil_enabled: True
workers: 8  iterations: 1000  warmup: 5  work_iters: 20000
===================

=== result ===
total_wall_s: 5.369269
avg_iter_s: 0.004226081
p50_iter_s: 0.004119016
p95_iter_s: 0.004885248

memory: (disabled)
==============
```

##### Profile memory

```console
✗ uv run --python 3.14+gil scenario1.py --mode processes --profile memory
=== environment ===
Python (cpython): 3.14.2 (main, Dec  9 2025, 19:03:28) [Clang 21.1.4 ]
OS: Linux-6.17.0-8-generic-x86_64-with-glibc2.42, arch x86_64
Mode: processes
Profile: memory
mp_start_method: forkserver
gil_enabled: True
workers: 8  iterations: 30  warmup: 2  work_iters: 20000
hold_ms: 1000  mem_sample: True  mem_interval_ms: 50
===================

=== result ===
total_wall_s: 30.805028
avg_iter_s: 1.024222078
p50_iter_s: 1.023542551
p95_iter_s: 1.029411792

--- memory (bytes) via psutil.memory_full_info() ---
rss_min: 28262400  rss_max: 180404224  rss_avg: 173159375
uss_min: 16506880  uss_max: 38678528  uss_avg: 37621174
pss_min: 19811328  pss_max: 52969472  pss_avg: 51388952
---------------------------------------------------
==============
```