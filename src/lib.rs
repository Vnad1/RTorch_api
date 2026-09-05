//! RTorch_api — safe Rust shell over the RTorch compute layer.
//!
//! This is the *safety vector* of the plugin: it wraps RTorch's public API
//! (`formula::run`, `rtw`, `device`) in Rust-owned, panic-safe types, and exposes
//! them through `ffi` as a stable C ABI for C++/Python/C#.
//!
//! RTorch stays compute-only. Nothing here modifies RTorch.

pub mod err;
pub mod ffi;
pub mod rule;

use std::panic::AssertUnwindSafe;

/// Convert a Rust panic into a C ABI error code, so a panic never crosses the
/// `extern "C"` boundary (which would be UB). Callers return the code.
pub(crate) fn ffi_guard<F: FnOnce() -> i32>(f: F) -> i32 {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(rc) => rc,
        Err(_) => {
            crate::err::set_last_error("rtorch_api: panic caught at C boundary");
            crate::err::RTORCH_API_E_RT
        }
    }
}

/// Like [`ffi_guard`] but for pointer-returning entry points: a panic yields a
/// null pointer (so the C caller can check for NULL), and the error is recorded.
pub(crate) fn ffi_guard_ptr<T, F: FnOnce() -> *mut T>(f: F) -> *mut T {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(p) => p,
        Err(_) => {
            crate::err::set_last_error("rtorch_api: panic caught at C boundary");
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These prove the panic-to-error contract works UNDER THE ACTIVE PANIC PROFILE.
    // With `panic = "unwind"` (the release profile), catch_unwind catches a panic
    // and ffi_guard converts it to RTORCH_API_E_RT. With `panic = "abort"` this
    // whole contract is void (catch_unwind is a no-op and the process aborts), so
    // this test would abort the harness — exactly the regression it guards.
    #[test]
    fn ffi_guard_catches_internal_panic() {
        let rc = ffi_guard(|| {
            panic!("intentional panic inside ffi_guard");
        });
        assert_eq!(rc, crate::err::RTORCH_API_E_RT, "panic must become E_RT");
        assert!(crate::err::last_error().contains("panic caught"));
    }

    #[test]
    fn ffi_guard_ptr_returns_null_on_panic() {
        let p: *mut u8 = ffi_guard_ptr(|| {
            panic!("intentional panic inside ffi_guard_ptr");
        });
        assert!(p.is_null(), "panic must yield a null pointer");
        assert!(crate::err::last_error().contains("panic caught"));
    }

    #[test]
    fn ffi_guard_ok_returns_code() {
        let rc = ffi_guard(|| 7);
        assert_eq!(rc, 7);
    }
}
