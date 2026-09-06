//! Rule system — plug-in rules added by users (community extensibility).
//!
//! A rule is a `name` -> callback. Users register a function (via C ABI
//! `rtorch_api_register_rule`, or directly here from Rust); the runtime finds it
//! by name and runs it. The registry is a **global `Mutex<HashMap>`** so a rule
//! registered on one thread is visible to (and runnable from) any other thread —
//! the cross-language (Python/C#) case where register and run may happen on
//! different threads.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// A rule callback. `in` blobs in, writes into `out`, `userdata` opaque.
/// Return 0 = ok, non-zero = error. Errored rules set last-error via the caller.
///
/// `Send`/`Sync` are required because the global registry (stored as `Arc`) is
/// shared across threads and a callback may be executed concurrently.
pub type RuleFn = Box<dyn Fn(&[crate::ffi::Blob], &mut crate::ffi::Blob, *mut std::ffi::c_void) -> i32 + Send + Sync>;

/// The C-ABI rule function pointer, matching `rtorch_api_rule_fn` in
/// `include/rtorch_api.h`: `int (*)(const blob* in, size_t n_in, blob* out,
/// void* userdata)`. Returns 0 = ok, non-zero = error. `rtorch_api_blob` and
/// [`crate::ffi::Blob`] share the same `#[repr(C)]` layout
/// (`{ const void* data; size_t len; }`).
pub type RuleCFn = unsafe extern "C" fn(*const crate::ffi::Blob, usize, *mut crate::ffi::Blob, *mut std::ffi::c_void) -> i32;

/// The rule registry: name -> callback. Global, cross-thread visible. We store an
/// `Arc` so `run` can clone the fn, release the registry lock, and execute the
/// callback *outside* the lock — a callback may re-enter the registry (register/
/// run) without deadlocking, and a long callback does not block other threads.
static RULES: OnceLock<Mutex<HashMap<String, Arc<RuleFn>>>> = OnceLock::new();

/// Re-export error bits used by ffi_guard / others.
pub use crate::err::{err, last_error, set_last_error, RTORCH_API_E_IO, RTORCH_API_E_NOMEM, RTORCH_API_E_PARAM, RTORCH_API_E_PARSE, RTORCH_API_E_RT, RTORCH_API_OK};

fn rules() -> &'static Mutex<HashMap<String, Arc<RuleFn>>> {
    RULES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A `Send` wrapper around a C-rule `userdata` raw pointer.
///
/// Safety contract: the caller (via `rtorch_api_register_rule`) must keep the
/// pointee alive for the rule's lifetime and ensure it is usable on any thread
/// that runs the rule — because the registry is global, a rule registered on one
/// thread may be run from another. This is the caller's obligation; the wrapper
/// only makes the pointer movable.
struct SendPtr(*mut std::ffi::c_void);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(&self) -> *mut std::ffi::c_void {
        self.0
    }
}

/// Register (or replace) a rule by name. `f` is moved into the registry.
pub fn register(name: &str, f: RuleFn) -> i32 {
    if name.is_empty() {
        return err(RTORCH_API_E_PARAM, "rtorch_api: rule name is empty");
    }
    match rules().lock() {
        Ok(mut r) => {
            r.insert(name.to_string(), Arc::new(f));
            RTORCH_API_OK
        }
        Err(_) => err(RTORCH_API_E_RT, "rtorch_api: rule registry lock poisoned"),
    }
}

/// Register a rule from a C function pointer. `cf` is called as
/// `cf(in, n_in, out, userdata)`; the stored `userdata` is passed through on
/// every invocation. Wrapped into a [`RuleFn`] so the C ABI (`rtorch_api_rule_fn`)
/// stays the single source of truth for community rules.
pub fn register_c(name: &str, cf: RuleCFn, userdata: *mut std::ffi::c_void) -> i32 {
    if name.is_empty() {
        return err(RTORCH_API_E_PARAM, "rtorch_api: rule name is empty");
    }
    let sd = SendPtr(userdata);
    let f: RuleFn = Box::new(move |inp, out, ud| {
        // Run-time userdata (from `rtorch_api_run_rule`) overrides the default
        // bound at register; null falls back to the registered context.
        let effective = if ud.is_null() { sd.get() } else { ud };
        // Safety: `cf` came from the caller for a pointer-sized fn, and the
        // caller contracted that it is valid. The C callback now returns an
        // `int rc` which we propagate up, so a C rule can report failures.
        unsafe { cf(inp.as_ptr(), inp.len(), out, effective) }
    });
    register(name, f)
}

/// Whether a rule is registered.
pub fn exists(name: &str) -> bool {
    rules().lock().map(|r| r.contains_key(name)).unwrap_or(false)
}

/// Run a registered rule. `userdata` is passed through. The registry lock is
/// released before the callback runs, so a callback may re-enter the registry and
/// long callbacks do not block other threads.
pub fn run(name: &str, inputs: &[crate::ffi::Blob], out: &mut crate::ffi::Blob, userdata: *mut std::ffi::c_void) -> i32 {
    // Clone the Arc under the lock, then drop the guard before executing.
    let entry = match rules().lock() {
        Ok(g) => match g.get(name) {
            Some(a) => Arc::clone(a),
            None => {
                return err(RTORCH_API_E_PARAM, &format!("rtorch_api: rule not found: {name}"));
            }
        },
        Err(PoisonError { .. }) => {
            return err(RTORCH_API_E_RT, "rtorch_api: rule registry lock poisoned");
        }
    };
    entry(inputs, out, userdata)
}

/// A test-friendly reset (not exposed over C ABI).
pub fn clear() {
    if let Ok(mut r) = rules().lock() {
        r.clear();
    }
}

