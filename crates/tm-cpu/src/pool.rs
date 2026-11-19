//! Flat-batch epoch-guarded spin-then-park thread pool for layer barriers.
//!
//! Design inspired by Strata `src/kernels/cpu/pool.cpp` (MIT License,
//! Copyright (c) 2026 Niko1221 and the Strata contributors):
//! - Each transformer/GDN layer is a natural barrier, so a flat batch dispatch
//!   avoids lock-free ring overhead.
//! - Monotonic `epoch` prevents late-waking workers from claiming work from a
//!   subsequent batch.
//! - Calling thread participates directly (`n_threads == 1` has zero sync cost).

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

type Trampoline = unsafe fn(*const (), usize, usize);

struct Shared {
    epoch: AtomicU64,
    shutdown: AtomicBool,
    total: AtomicUsize,
    chunk: AtomicUsize,
    next_idx: AtomicUsize,
    done_workers: AtomicUsize,
    work_fn: AtomicPtr<()>,
    work_ctx: AtomicPtr<()>,
    wake_mu: Mutex<u64>,
    wake_cv: Condvar,
}

/// Reusable thread pool sized to physical performance cores.
pub struct ThreadPool {
    n_threads: usize,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Creates a pool with `n_threads` total execution threads (including the caller).
    pub fn new(n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            total: AtomicUsize::new(0),
            chunk: AtomicUsize::new(1),
            next_idx: AtomicUsize::new(0),
            done_workers: AtomicUsize::new(0),
            work_fn: AtomicPtr::new(core::ptr::null_mut()),
            work_ctx: AtomicPtr::new(core::ptr::null_mut()),
            wake_mu: Mutex::new(0),
            wake_cv: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(n_threads - 1);
        for _ in 1..n_threads {
            let s = Arc::clone(&shared);
            workers.push(thread::spawn(move || worker_loop(s)));
        }

        Self {
            n_threads,
            shared,
            workers,
        }
    }

    pub fn num_threads(&self) -> usize {
        self.n_threads
    }

    /// Executes `f(start, end)` in parallel across `[0, total)` with chunk size `chunk`.
    /// Blocks until all chunks and all workers have completed the current epoch.
    pub fn parallel_for<F>(&self, total: usize, chunk: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total == 0 {
            return;
        }
        let chunk = chunk.max(1);
        if self.n_threads == 1 || total <= chunk {
            f(0, total);
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize) + Sync>(ctx: *const (), start: usize, end: usize) {
            // SAFETY: `ctx` points to the caller's stack-allocated `F` which remains
            // alive and immutable for the entire duration of `parallel_for` (we join
            // via `done_workers == n_workers` before returning).
            let func = unsafe { &*(ctx as *const F) };
            func(start, end);
        }

        let fn_ptr: Trampoline = trampoline::<F>;
        let ctx_ptr = (&f as *const F) as *mut ();

        let s = &self.shared;
        s.work_fn.store(fn_ptr as *mut (), Ordering::Relaxed);
        s.work_ctx.store(ctx_ptr, Ordering::Relaxed);
        s.total.store(total, Ordering::Relaxed);
        s.chunk.store(chunk, Ordering::Relaxed);
        s.next_idx.store(0, Ordering::Relaxed);
        s.done_workers.store(0, Ordering::Relaxed);

        let next_epoch = s.epoch.load(Ordering::Relaxed) + 1;
        {
            let mut guard = s.wake_mu.lock().unwrap_or_else(|e| e.into_inner());
            *guard = next_epoch;
            s.epoch.store(next_epoch, Ordering::Release);
            s.wake_cv.notify_all();
        }

        // Caller thread also claims chunks.
        loop {
            let start = s.next_idx.fetch_add(chunk, Ordering::Relaxed);
            if start >= total {
                break;
            }
            let end = (start + chunk).min(total);
            f(start, end);
        }

        // Wait until all background workers have exited this epoch.
        let n_workers = self.workers.len();
        while s.done_workers.load(Ordering::Acquire) < n_workers {
            core::hint::spin_loop();
        }
    }
}

fn worker_loop(s: Arc<Shared>) {
    let mut seen_epoch = 0u64;

    loop {
        // Brief spin before parking on Condvar to minimize layer-barrier latency.
        let mut cur_epoch = s.epoch.load(Ordering::Acquire);
        if cur_epoch == seen_epoch {
            for _ in 0..4000 {
                if s.shutdown.load(Ordering::Acquire) {
                    return;
                }
                cur_epoch = s.epoch.load(Ordering::Acquire);
                if cur_epoch != seen_epoch {
                    break;
                }
                core::hint::spin_loop();
            }
        }

        if cur_epoch == seen_epoch {
            let mut guard = s.wake_mu.lock().unwrap_or_else(|e| e.into_inner());
            while *guard == seen_epoch && !s.shutdown.load(Ordering::Acquire) {
                guard = s.wake_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
            if s.shutdown.load(Ordering::Acquire) {
                return;
            }
            cur_epoch = s.epoch.load(Ordering::Acquire);
        }

        if s.shutdown.load(Ordering::Acquire) {
            return;
        }

        seen_epoch = cur_epoch;

        let total = s.total.load(Ordering::Relaxed);
        let chunk = s.chunk.load(Ordering::Relaxed);
        let fn_raw = s.work_fn.load(Ordering::Relaxed);
        let ctx = s.work_ctx.load(Ordering::Relaxed) as *const ();

        // SAFETY: `fn_raw` was stored by `parallel_for` as a valid `Trampoline`
        // before the Release store to `epoch` that we just observed with Acquire.
        let func: Trampoline = unsafe { core::mem::transmute(fn_raw) };

        loop {
            let start = s.next_idx.fetch_add(chunk, Ordering::Relaxed);
            if start >= total {
                break;
            }
            let end = (start + chunk).min(total);
            // SAFETY: `ctx` is valid until we increment `done_workers` below.
            unsafe {
                func(ctx, start, end);
            }
        }

        s.done_workers.fetch_add(1, Ordering::Release);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        {
            let _guard = self
                .shared
                .wake_mu
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.shared.wake_cv.notify_all();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}
