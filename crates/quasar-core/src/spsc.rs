//! Lock-free single-producer / single-consumer ring queue.
//!
//! Used to carry configuration commands from the compute / API thread to the audio thread, and
//! retired DSP state back the other way, without either side ever taking a lock (#75).
//!
//! * The ring is preallocated once ([`spsc_channel`]); `push` / `pop` never allocate, lock or
//!   wait, and are wait-free (a bounded number of atomic operations).
//! * Exactly one [`SpscProducer`] and one [`SpscConsumer`] exist per queue (neither is `Clone`,
//!   and both take `&mut self`), which is what makes the unsynchronised slot access sound.
//! * Capacity is rounded up to a power of two (minimum 2). `push` on a full queue hands the value
//!   back instead of blocking.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// An atomic index on its own cache line (avoids false sharing between the two threads).
#[repr(align(64))]
struct PaddedIndex(AtomicUsize);

struct Inner<T> {
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
    /// Next slot to read (written by the consumer only).
    head: PaddedIndex,
    /// Next slot to write (written by the producer only).
    tail: PaddedIndex,
}

// SAFETY: a slot is only ever touched by the producer (while empty, before `tail` is advanced
// past it) or by the consumer (while full, before `head` is advanced past it); the acquire /
// release pairs on `head` / `tail` order those accesses. Values move between threads, so `T: Send`.
unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let mut head = *self.head.0.get_mut();
        let tail = *self.tail.0.get_mut();
        while head != tail {
            // SAFETY: slots in [head, tail) are initialised and owned by the queue.
            unsafe { (*self.slots[head & self.mask].get()).assume_init_drop() };
            head = head.wrapping_add(1);
        }
    }
}

/// Producer half of an SPSC queue.
pub struct SpscProducer<T> {
    inner: Arc<Inner<T>>,
}

/// Consumer half of an SPSC queue.
pub struct SpscConsumer<T> {
    inner: Arc<Inner<T>>,
}

/// Create a queue holding up to `capacity` items (rounded up to a power of two, at least 2).
pub fn spsc_channel<T: Send>(capacity: usize) -> (SpscProducer<T>, SpscConsumer<T>) {
    let cap = capacity.max(2).next_power_of_two();
    let slots: Box<[UnsafeCell<MaybeUninit<T>>]> =
        (0..cap).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
    let inner = Arc::new(Inner {
        slots,
        mask: cap - 1,
        head: PaddedIndex(AtomicUsize::new(0)),
        tail: PaddedIndex(AtomicUsize::new(0)),
    });
    (SpscProducer { inner: Arc::clone(&inner) }, SpscConsumer { inner })
}

impl<T> SpscProducer<T> {
    /// Capacity of the ring.
    pub fn capacity(&self) -> usize {
        self.inner.mask + 1
    }

    /// Push `value`; returns it back as `Err` if the queue is full. Wait-free, never allocates.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        let tail = self.inner.tail.0.load(Ordering::Relaxed);
        let head = self.inner.head.0.load(Ordering::Acquire);
        if tail.wrapping_sub(head) > self.inner.mask {
            return Err(value);
        }
        // SAFETY: the slot at `tail` is empty (the consumer has moved past it) and only this
        // producer writes it before publishing `tail + 1`.
        unsafe { (*self.inner.slots[tail & self.inner.mask].get()).write(value) };
        self.inner.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Items currently queued (a snapshot; the consumer may be popping concurrently).
    pub fn len(&self) -> usize {
        let tail = self.inner.tail.0.load(Ordering::Relaxed);
        tail.wrapping_sub(self.inner.head.0.load(Ordering::Acquire))
    }

    /// Whether the queue is (momentarily) empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> SpscConsumer<T> {
    /// Pop the oldest item, or `None` if the queue is empty. Wait-free, never allocates.
    pub fn pop(&mut self) -> Option<T> {
        let head = self.inner.head.0.load(Ordering::Relaxed);
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        // SAFETY: the slot at `head` was initialised by the producer before it published `tail`
        // past it, and only this consumer reads it before publishing `head + 1`.
        let value = unsafe { (*self.inner.slots[head & self.inner.mask].get()).assume_init_read() };
        self.inner.head.0.store(head.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Items currently queued (a snapshot; the producer may be pushing concurrently).
    pub fn len(&self) -> usize {
        let head = self.inner.head.0.load(Ordering::Relaxed);
        self.inner.tail.0.load(Ordering::Acquire).wrapping_sub(head)
    }

    /// Whether the queue is (momentarily) empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
