//! Boundary / panic probe: every C ABI entry is fed hostile inputs (null, empty,
//! out-of-range indices, garbage bytes) and must NOT panic and NOT read/write OOB.
//! We wrap each call in `catch_unwind` so a panic — which would be a UB risk at the
//! FFI boundary — is caught and reported as a failure. This complements the
//! happy-path smoke tests (ffi_smoke.rs).

use rtorch_for_models_api::ffi::*;

use std::panic::{catch_unwind, AssertUnwindSafe};

fn no_panic(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_ok()
}

#[test]
fn tensor_hostile_args_no_panic() {
    // Garbage/large dims (overflow in product) must not panic.
    assert!(no_panic(|| {
        let bad = [usize::MAX, 2];
        let t = rtorch_api_tensor_new(bad.as_ptr(), 2, 0);
        // numel = usize::MAX * 2 overflows; the crate may panic on product.
        let _ = t;
    }));
    // Null dims with rank 0 is allowed (empty tensor), must be non-null.
    let t = rtorch_api_tensor_new(std::ptr::null(), 0, 0);
    assert!(!t.is_null());
    assert_eq!(rtorch_api_tensor_numel(t), 0);
    rtorch_api_tensor_free(t);
    // Freeing null is a no-op.
    no_panic(|| rtorch_api_tensor_free(std::ptr::null_mut()));
}

#[test]
fn tensor_product_overflow_is_handled() {
    // `dims.iter().product()` on huge dims would overflow usize; the safe shell
    // must reject it (return null + error), never attempt a huge allocation or
    // overflow-panic across the boundary.
    let huge = [usize::MAX / 2, usize::MAX / 2];
    let t = rtorch_api_tensor_new(huge.as_ptr(), 2, 0);
    assert!(t.is_null(), "overflowing dims must be rejected (null)");
    // A non-overflowing but absurd dim set is also capped (returns null).
    let big = [1usize << 31, 1usize << 31]; // 2^62, overflows
    assert!(rtorch_api_tensor_new(big.as_ptr(), 2, 0).is_null());
    let absurd = [u32::MAX as usize, u32::MAX as usize]; // ~1.8e19, overflows
    assert!(rtorch_api_tensor_new(absurd.as_ptr(), 2, 0).is_null());
}

#[test]
fn session_hostile_args_no_panic() {
    // Null out (session_execute with null out) -> E_PARAM, no panic.
    assert!(no_panic(|| {
        let rc = rtorch_api_session_execute(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null_mut(), 0);
        assert_eq!(rc, 1); // E_PARAM
    }));
    // Null session but valid out.
    let mut out_bytes = [0u8; 16];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 16 };
    assert!(no_panic(|| {
        let rc = rtorch_api_session_execute(std::ptr::null_mut(), std::ptr::null(), 0, &mut out_blob, 0);
        assert_eq!(rc, 1);
    }));
}

#[test]
fn rtw_decode_garbage_no_panic() {
    // Garbage bytes that aren't an RTW -> E_PARSE/null, no panic.
    assert!(no_panic(|| {
        let junk = [0xAAu8; 16];
        let r = rtorch_api_rtw_decode(junk.as_ptr(), junk.len());
        assert!(r.is_null());
    }));
    // Null bytes with len 0 -> decode fails -> null, no panic.
    assert!(no_panic(|| {
        let r = rtorch_api_rtw_decode(std::ptr::null(), 0);
        let _ = r;
    }));
}

#[test]
fn model_decode_garbage_no_panic() {
    // Model payload that's not a valid encode_model payload -> null, no panic.
    assert!(no_panic(|| {
        let junk = [0xFFu8; 8];
        let m = rtorch_api_model_decode(junk.as_ptr(), junk.len());
        assert!(m.is_null());
    }));
}

#[test]
fn memory_decode_garbage_no_panic() {
    assert!(no_panic(|| {
        let junk = [0xFFu8; 8];
        let m = rtorch_api_memory_decode(junk.as_ptr(), junk.len());
        assert!(m.is_null());
    }));
}
