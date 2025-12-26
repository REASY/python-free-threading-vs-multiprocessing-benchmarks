// Scenario 2 (Rust): Shared global unique set with a lock (threads only)
//
// This is a Rust version of the "threads" mode from `scenario2_shared_unique_set.py`.
// The goal is to estimate the overhead of the Python runtime by comparing against a
// compiled implementation of the same high-level algorithm.
//
// Build (recommended):
//   rustc -O -C target-cpu=native -C opt-level=3 scenario2_shared_unique_set.rs
//
// Run:
//   ./scenario2_shared_unique_set --workers 8 --duration-ms 5000 --id-space 10000000
//
// Output format intentionally mirrors the Python script.

use std::collections::HashSet;
use std::env;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Fallback for non-u64 keys; not used in this benchmark.
        let mut h = 0xcbf29ce484222325u64; // FNV-1a offset basis
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.0 = h;
    }

    fn write_u64(&mut self, i: u64) {
        // CPython's hash for small ints is effectively the integer value.
        self.0 = i;
    }
}

type FastBuildHasher = BuildHasherDefault<IdentityHasher>;
type SharedSet = HashSet<u64, FastBuildHasher>;

#[derive(Clone, Copy, Debug, Default)]
struct WorkerStats {
    attempts: u64,
    inserts: u64,
    lock_wait_ns: u64,
}

impl WorkerStats {
    fn add(&mut self, other: WorkerStats) {
        self.attempts = self.attempts.wrapping_add(other.attempts);
        self.inserts = self.inserts.wrapping_add(other.inserts);
        self.lock_wait_ns = self.lock_wait_ns.wrapping_add(other.lock_wait_ns);
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    workers: usize,
    warmup: usize,
    duration: Duration,
    id_space: u64,
}

#[derive(Clone, Copy, Debug)]
struct RunResult {
    wall: Duration,
    stats: WorkerStats,
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn usage() -> String {
    let exe = env::args().next().unwrap_or_else(|| "scenario2_shared_unique_set".into());
    format!(
        "\
Usage: {exe} [--workers N] [--warmup N] [--duration-ms MS] [--id-space N]

Defaults:
  --workers 8
  --warmup  1
  --duration-ms 5000
  --id-space 10000000
"
    )
}

fn parse_args() -> Result<Config, String> {
    let mut workers: usize = 8;
    let mut warmup: usize = 1;
    let mut duration_ms: u64 = 5000;
    let mut id_space: u64 = 10_000_000;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Err(usage()),
            "--workers" => {
                i += 1;
                workers = args
                    .get(i)
                    .ok_or_else(|| "Missing value for --workers".to_string())?
                    .parse()
                    .map_err(|_| "Invalid integer for --workers".to_string())?;
            }
            "--warmup" => {
                i += 1;
                warmup = args
                    .get(i)
                    .ok_or_else(|| "Missing value for --warmup".to_string())?
                    .parse()
                    .map_err(|_| "Invalid integer for --warmup".to_string())?;
            }
            "--duration-ms" => {
                i += 1;
                duration_ms = args
                    .get(i)
                    .ok_or_else(|| "Missing value for --duration-ms".to_string())?
                    .parse()
                    .map_err(|_| "Invalid integer for --duration-ms".to_string())?;
            }
            "--id-space" => {
                i += 1;
                id_space = args
                    .get(i)
                    .ok_or_else(|| "Missing value for --id-space".to_string())?
                    .parse()
                    .map_err(|_| "Invalid integer for --id-space".to_string())?;
            }
            other => return Err(format!("Unknown argument: {other}\n\n{}", usage())),
        }
        i += 1;
    }

    if workers == 0 {
        return Err("--workers must be > 0".to_string());
    }
    if duration_ms == 0 {
        return Err("--duration-ms must be > 0".to_string());
    }
    if id_space == 0 {
        return Err("--id-space must be > 0".to_string());
    }

    Ok(Config {
        workers,
        warmup,
        duration: Duration::from_millis(duration_ms),
        id_space,
    })
}

fn run_once(cfg: Config) -> RunResult {
    let shared_set: Arc<Mutex<SharedSet>> = Arc::new(Mutex::new(SharedSet::default()));
    let barrier = Arc::new(Barrier::new(cfg.workers + 1));
    let start_pair: Arc<(Mutex<Option<Instant>>, Condvar)> =
        Arc::new((Mutex::new(None), Condvar::new()));

    let it0 = Instant::now();
    let mut handles = Vec::with_capacity(cfg.workers);

    for worker_index in 0..cfg.workers {
        let shared_set = Arc::clone(&shared_set);
        let barrier = Arc::clone(&barrier);
        let start_pair = Arc::clone(&start_pair);
        let id_space = cfg.id_space;
        let duration = cfg.duration;

        handles.push(thread::spawn(move || -> WorkerStats {
            let seed = 0xC0FFEEu64 ^ ((worker_index as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = SplitMix64::new(seed);

            // Fair start: ensure all threads are spawned and parked before starting the timed loop.
            barrier.wait();

            // Separate start signal so the main thread controls the "t=0" instant.
            let start = {
                let (lock, cv) = &*start_pair;
                let mut guard = lock.lock().unwrap();
                while guard.is_none() {
                    guard = cv.wait(guard).unwrap();
                }
                guard.expect("start instant must be set")
            };

            let end = start + duration;
            let mut attempts: u64 = 0;
            let mut inserts: u64 = 0;
            let mut lock_wait_ns: u64 = 0;

            while Instant::now() < end {
                let uid = rng.next_u64() % id_space;

                let t0 = Instant::now();
                let mut guard = shared_set.lock().unwrap();
                let t1 = Instant::now();
                lock_wait_ns = lock_wait_ns.wrapping_add(t1.duration_since(t0).as_nanos() as u64);

                // Critical section: check + insert (same shape as the Python version).
                if !guard.contains(&uid) {
                    guard.insert(uid);
                    inserts += 1;
                }
                drop(guard);

                attempts += 1;
            }

            WorkerStats {
                attempts,
                inserts,
                lock_wait_ns,
            }
        }));
    }

    barrier.wait();
    let start = Instant::now();
    {
        let (lock, cv) = &*start_pair;
        let mut guard = lock.lock().unwrap();
        *guard = Some(start);
        cv.notify_all();
    }

    let mut total = WorkerStats::default();
    for h in handles {
        total.add(h.join().expect("worker thread panicked"));
    }

    RunResult {
        wall: it0.elapsed(),
        stats: total,
    }
}

fn main() {
    let cfg = match parse_args() {
        Ok(cfg) => cfg,
        Err(msg) => {
            // Treat `--help` as a non-error path (it returns usage text).
            if msg.starts_with("Usage:") {
                print!("{msg}");
                return;
            }
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    println!("=== environment ===");
    println!("OS: {}  arch: {}", env::consts::OS, env::consts::ARCH);
    println!("Mode: threads (Rust)");
    println!(
        "workers: {}  warmup: {}  duration_ms: {}  id_space: {}",
        cfg.workers,
        cfg.warmup,
        cfg.duration.as_millis(),
        cfg.id_space
    );
    println!("===================");

    for _ in 0..cfg.warmup {
        let _ = run_once(cfg);
    }

    let res = run_once(cfg);

    let wall_s = res.wall.as_secs_f64().max(1e-12);
    let attempts = res.stats.attempts;
    let inserts = res.stats.inserts;
    let dup_rate = if attempts == 0 {
        0.0
    } else {
        (1.0 - (inserts as f64 / attempts as f64)) * 100.0
    };
    let ops_s = attempts as f64 / wall_s;
    let inserts_s = inserts as f64 / wall_s;
    let avg_lock_wait_ns = if attempts == 0 {
        0.0
    } else {
        res.stats.lock_wait_ns as f64 / attempts as f64
    };

    println!("\n=== result ===");
    println!("wall_s: {:.6}", wall_s);
    println!(
        "attempts: {}  inserts: {}  dup_rate_pct: {:.2}",
        attempts, inserts, dup_rate
    );
    println!("ops_per_s: {:.2}", ops_s);
    println!("inserts_per_s: {:.2}", inserts_s);
    println!("avg_lock_wait_ns: {:.1}", avg_lock_wait_ns);
    println!("==============\n");
}

