//! Shared test harness: a counting global allocator.
//!
//! Each integration-test file that does `mod common;` becomes its own binary, so the
//! `#[global_allocator]` below only affects those binaries. Allocations are counted PER THREAD
//! (a const-initialised thread-local, which itself never allocates), so parallel tests and the
//! libtest harness threads do not disturb a measurement.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static FREES: Cell<u64> = const { Cell::new(0) };
}

pub struct CountingAlloc;

#[inline]
fn bump() {
    // `try_with`: thread-local may be torn down during thread exit.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        System.realloc(p, l, n)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        let _ = FREES.try_with(|c| c.set(c.get() + 1));
        System.dealloc(p, l)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Number of heap allocations (alloc / alloc_zeroed / realloc) made by the calling thread so far.
pub fn thread_allocs() -> u64 {
    ALLOCS.with(|c| c.get())
}

/// Number of heap frees made by the calling thread so far.
pub fn thread_frees() -> u64 {
    FREES.with(|c| c.get())
}

/// Run `f` and return `(allocations, frees)` made by the calling thread inside it.
pub fn count_alloc_free<R>(f: impl FnOnce() -> R) -> (R, u64, u64) {
    let (a0, f0) = (thread_allocs(), thread_frees());
    let r = f();
    (r, thread_allocs() - a0, thread_frees() - f0)
}

/// Run `f` and return how many allocations the calling thread made inside it.
pub fn count_allocs<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = thread_allocs();
    let r = f();
    (r, thread_allocs() - before)
}
