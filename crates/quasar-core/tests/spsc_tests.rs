//! Lock-free SPSC ring: ordering, capacity, drop semantics and a two-thread stress test.

use quasar_core::spsc::spsc_channel;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn fifo_order_and_full_empty() {
    let (mut tx, mut rx) = spsc_channel::<u32>(4);
    assert_eq!(tx.capacity(), 4);
    assert!(rx.pop().is_none());
    for i in 0..4 {
        assert!(tx.push(i).is_ok());
    }
    assert_eq!(tx.push(99), Err(99), "full queue hands the value back");
    assert_eq!(rx.len(), 4);
    for i in 0..4 {
        assert_eq!(rx.pop(), Some(i));
    }
    assert!(rx.pop().is_none());
    // Wrap-around many times.
    for round in 0..100u32 {
        for i in 0..3 {
            tx.push(round * 10 + i).unwrap();
        }
        for i in 0..3 {
            assert_eq!(rx.pop(), Some(round * 10 + i));
        }
    }
}

#[test]
fn capacity_rounds_up_to_power_of_two() {
    let (tx, _rx) = spsc_channel::<u8>(5);
    assert_eq!(tx.capacity(), 8);
    let (tx, _rx) = spsc_channel::<u8>(0);
    assert_eq!(tx.capacity(), 2);
}

#[test]
fn queued_items_are_dropped_with_the_queue() {
    struct D(Arc<AtomicUsize>);
    impl Drop for D {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    {
        let (mut tx, mut rx) = spsc_channel::<D>(8);
        for _ in 0..5 {
            assert!(tx.push(D(drops.clone())).is_ok());
        }
        drop(rx.pop()); // one popped and dropped
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 5, "the 4 left in the queue are dropped with it");
}

/// Two threads: the producer pushes a long counting sequence (spinning when full), the consumer
/// checks every value arrives exactly once and in order, with boxed payloads to catch tearing.
#[test]
fn two_thread_stress_preserves_order_and_content() {
    const N: u64 = 2_000_000;
    let (mut tx, mut rx) = spsc_channel::<Box<(u64, u64)>>(64);
    let producer = std::thread::spawn(move || {
        for i in 0..N {
            let mut v = Box::new((i, !i));
            loop {
                match tx.push(v) {
                    Ok(()) => break,
                    Err(back) => {
                        v = back;
                        std::hint::spin_loop();
                    }
                }
            }
        }
    });
    let consumer = std::thread::spawn(move || {
        let mut next = 0u64;
        while next < N {
            match rx.pop() {
                Some(b) => {
                    assert_eq!(b.0, next, "out of order / lost / duplicated");
                    assert_eq!(b.1, !next, "payload torn");
                    next += 1;
                }
                None => std::hint::spin_loop(),
            }
        }
        assert!(rx.pop().is_none());
    });
    producer.join().unwrap();
    consumer.join().unwrap();
}
