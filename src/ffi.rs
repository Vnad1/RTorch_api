//! The `extern "C"` boundary of RTorch_api. Every symbol here matches
//! `include/rtorch_api.h`. All functions are panic-safe (see `ffi_guard`) and
//! do explicit bounds/pointer checks before touching memory.
//!
//! These are safe `extern "C"` entry points (not `unsafe fn`) so the C ABI stays
//! trivially callable; each checks its raw-pointer arguments internally. clippy
//! flags the internal dereference on a safe fn, which is exactly the intended
//! pattern here, so we allow `not_unsafe_ptr_arg_deref` for the module.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::err::{err, last_error, RTORCH_API_E_NOMEM, RTORCH_API_E_PARAM, RTORCH_API_E_PARSE, RTORCH_API_OK};
use crate::ffi_guard;
use crate::ffi_guard_ptr;

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// The C ABI byte blob (mirrors `rtorch_api_blob`). Copy so we can snapshot the
/// input array passed by the C caller without aliasing.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Blob {
    pub data: *const std::ffi::c_void,
    pub len: usize,
}

// ---------------------------------------------------------------------------
// Tensor.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct OpaqueTensor {
    dims: Vec<usize>,
    data: Vec<f32>,
    dtype: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_new(dims: *const usize, rank: usize, dtype: i32) -> *mut OpaqueTensor {
    ffi_guard_ptr(|| {
        if dims.is_null() && rank > 0 {
            err(RTORCH_API_E_PARAM, "rtorch_api: null dims with rank>0");
            return std::ptr::null_mut();
        }
        let dims = if rank == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(dims, rank).to_vec() }
        };
        // Only F32 is actually backed by the safe shell (`data: Vec<f32>` and
        // `tensor_data` returns `const float*`). Reject F64/I32 rather than
        // silently storing them as f32 (which would let a caller read the wrong
        // width and overrun — a real memory-safety bug).
        if dtype != 0 {
            err(RTORCH_API_E_PARAM, "rtorch_api: only RTORCH_API_DTYPE_F32 supported");
            return std::ptr::null_mut();
        }
        // Compute numel with overflow checking: the product of dims may overflow
        // (hostile caller passes huge dims), which would otherwise make `vec![…]
        // numel` allocate a massive block (or overflow-panic in debug) and abort
        // the process. Reject any overflow / absurd size with a proper error.
        // A 0-rank tensor has 0 elements.
        let mut numel: usize = if dims.is_empty() { 0 } else { 1 };
        for &d in &dims {
            numel = match numel.checked_mul(d) {
                Some(v) => v,
                None => {
                    err(RTORCH_API_E_PARAM, "rtorch_api: tensor numel overflows usize");
                    return std::ptr::null_mut();
                }
            };
            // Optional sanity cap so a bogus-but-non-overflowing dim set doesn't
            // attempt a multi-GB allocation before failing.
            if numel > (1 << 29) {
                err(RTORCH_API_E_NOMEM, "rtorch_api: tensor too large");
                return std::ptr::null_mut();
            }
        }
        let data = vec![0.0f32; numel];
        Box::into_raw(Box::new(OpaqueTensor { dims, data, dtype }))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        unsafe { drop(Box::from_raw(t)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_data(t: *const OpaqueTensor) -> *const f32 {
    if t.is_null() {
        return std::ptr::null();
    }
    let t = unsafe { &*t };
    t.data.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_numel(t: *const OpaqueTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    unsafe { (*t).data.len() }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_rank(t: *const OpaqueTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    unsafe { (*t).dims.len() }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_dim(t: *const OpaqueTensor, i: usize) -> usize {
    if t.is_null() {
        return 0;
    }
    let t = unsafe { &*t };
    t.dims.get(i).copied().unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_tensor_dtype(t: *const OpaqueTensor) -> i32 {
    if t.is_null() {
        return -1;
    }
    unsafe { (*t).dtype }
}

// ---------------------------------------------------------------------------
// Formula session 鈥?compile once / execute many.
// ---------------------------------------------------------------------------

pub struct OpaqueSession {
    // A session keeps a *compiled* DLL path (never the raw .cpp). For a `.cpp`
    // input we compile it once at compile() time; for a `.dll` input we just
    // cache the resolved path. execute() then calls rtorch's formula::run with
    // that DLL, so a `.cpp` is compiled exactly once per session (M2 fixes the
    // previous "recompile every execute").
    //
    // The persistent LoadLibrary handle (HINSTANCE kept open across executes)
    // is a follow-up refinement; here we already remove the dominant cost —
    // the per-call g++ compile.
    dll: PathBuf,
    // True if we compiled `dll` ourselves (a temp DLL we own and should clean up).
    owns_dll: bool,
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_session_compile(formula_src: *const std::ffi::c_char, ref_dir: *const std::ffi::c_char) -> *mut OpaqueSession {
    ffi_guard_ptr(|| {
        if formula_src.is_null() {
            err(RTORCH_API_E_PARAM, "rtorch_api: null formula_src");
            return std::ptr::null_mut();
        }
        let src = unsafe { std::ffi::CStr::from_ptr(formula_src) }.to_string_lossy().to_string();
        let ref_dir = if ref_dir.is_null() {
            None
        } else {
            let s = unsafe { std::ffi::CStr::from_ptr(ref_dir) }.to_string_lossy().to_string();
            if s.is_empty() { None } else { Some(s) }
        };
        let src_path = PathBuf::from(&src);

        // Deterministic per-session temp DLL name so concurrent sessions don't clash.
        static CTR: AtomicU64 = AtomicU64::new(0);
        let session_id = CTR.fetch_add(1, Ordering::Relaxed);
        let out_dll = env::current_dir()
            .ok()
            .unwrap_or_else(env::temp_dir)
            .join(format!("rtorch_api_rt_{}_{}.dll", std::process::id(), session_id));

        let (dll, owns_dll) = if is_dll(&src_path) {
            // Pre-built DLL: absolute path (Windows' default search omits cwd).
            let abs = std::fs::canonicalize(&src_path).unwrap_or(src_path);
            (abs, false)
        } else {
            match compile_once(&src_path, ref_dir.as_deref(), &out_dll) {
                Ok(()) => (out_dll, true),
                Err(msg) => {
                    err(RTORCH_API_E_PARSE, &format!("rtorch_api: compile failed: {msg}"));
                    return std::ptr::null_mut();
                }
            }
        };
        Box::into_raw(Box::new(OpaqueSession { dll, owns_dll }))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_session_free(s: *mut OpaqueSession) {
    if !s.is_null() {
        let s = unsafe { Box::from_raw(s) };
        if s.owns_dll {
            let _ = std::fs::remove_file(&s.dll);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_session_execute(
    s: *mut OpaqueSession,
    in_blobs: *const Blob,
    n_in: usize,
    out: *mut Blob,
    device: i32,
) -> i32 {
    ffi_guard(|| {
        if s.is_null() || out.is_null() {
            return err(RTORCH_API_E_PARAM, "rtorch_api: null session/out");
        }
        if in_blobs.is_null() && n_in > 0 {
            return err(RTORCH_API_E_PARAM, "rtorch_api: null in_blobs");
        }
        let s = unsafe { &*s };
        let blobs = if n_in == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(in_blobs, n_in) }.to_vec()
        };
        // materialize each input into bytes
        let mut inputs: Vec<Vec<u8>> = Vec::new();
        for b in &blobs {
            if b.data.is_null() && b.len > 0 {
                return err(RTORCH_API_E_PARAM, "rtorch_api: null input data");
            }
            let slice = if b.len == 0 {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(b.data as *const u8, b.len) }
            };
            inputs.push(slice.to_vec());
        }
        // formula::run loads + dispatches + unloads. The persistent handle (kept
        // open across executes) is a follow-up; here the cached DLL means no
        // recompile per call — the only per-call cost is load/dispatch.
        let res = rtorch::formula::run(Path::new(&s.dll), &[], &inputs, device);
        match res {
            Ok(bytes) => {
                let out = unsafe { &mut *out };
                if bytes.len() > out.len {
                    // write what fits, report truncation as a parse-ish error
                    return err(RTORCH_API_E_PARAM, "rtorch_api: output buffer too small");
                }
                if !out.data.is_null() && !bytes.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.data as *mut u8, bytes.len());
                    }
                }
                out.len = bytes.len();
                RTORCH_API_OK
            }
            Err(e) => err(RTORCH_API_E_PARSE, &format!("rtorch_api: formula run failed: {e}")),
        }
    })
}

// ---------------------------------------------------------------------------
// Compile a formula `.cpp` once into a DLL (same flags as RTorch's pipeline so
// the plugin produces the same binary it would; only difference is we keep the
// DLL alive for the session instead of deleting after one run).
// ---------------------------------------------------------------------------

fn is_dll(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("dll"))
        .unwrap_or(false)
}

fn compile_once(formula: &Path, ref_dir: Option<&str>, out_dll: &Path) -> Result<(), String> {
    let compiler = rtorch::formula::find_compiler()
        .ok_or_else(|| "no C++ compiler found (need MinGW g++; set RTORCH_GXX)".to_string())?;
    // Ensure compiler runtime DLLs are on PATH so the resulting DLL (which we
    // LoadLibrary later) can resolve libwinpthread etc.
    if let Some(dir) = compiler.parent() {
        let path = env::var("PATH").unwrap_or_default();
        let dstr = dir.display().to_string();
        if !path.split(';').any(|p| p.eq_ignore_ascii_case(&dstr)) {
            let newpath = format!("{dstr};{path}");
            unsafe { env::set_var("PATH", newpath); }
        }
    }

    let _ = std::fs::remove_file(out_dll);
    let mut cmd = Command::new(&compiler);
    cmd.arg("-shared")
        .arg("-fPIC")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-march=native")
        .arg("-ffast-math")
        .arg("-funroll-loops")
        .arg("-static")
        .arg("-static-libgcc")
        .arg("-static-libstdc++")
        .arg(formula);
    // include formula dir + refs dir so `#include "rtorch.h"` resolves.
    if let Some(dir) = formula.parent() {
        cmd.arg("-I").arg(dir);
    }
    if let Some(rd) = ref_dir {
        cmd.arg("-I").arg(rd);
    }
    cmd.arg("-o").arg(out_dll);

    let st = cmd.status().map_err(|e| format!("spawn g++: {e}"))?;
    if !st.success() {
        let _ = std::fs::remove_file(out_dll);
        return Err("compilation failed (see g++ output above)".to_string());
    }
    if !out_dll.exists() {
        return Err("g++ reported success but no DLL produced".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RTW container / model / memory 鈥?first pass (M3 will add real accessors).
// ---------------------------------------------------------------------------

pub struct OpaqueRtw {
    kind: u8,
    bytes: Vec<u8>,
    // The RTW self-describing Manifest (artifact_id / location / format_version),
    // exposed so cross-language consumers can see "which library / where / the
    // RTW format version" the same way RTorch does. None for a legacy no-manifest
    // RTW.
    manifest: Option<rtorch::rtw::Manifest>,
}
pub struct OpaqueModel {
    // A decoded `rtorch::rtw::Model` (model payload = the bytes after the RTW
    // header for kind=KIND_MODEL). The accessors read directly from this.
    model: rtorch::rtw::Model,
}
pub struct OpaqueMemory {
    memory: rtorch::rtw::Memory,
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_last_error() -> *const std::ffi::c_char {
    // Return a pointer to a thread-local, NUL-terminated copy that lives until
    // the next call on the same thread. No leak: each call replaces the buffer.
    use std::cell::RefCell;
    thread_local! {
        static LAST_ERR_C: RefCell<Option<std::ffi::CString>> = const { RefCell::new(None) };
    }
    let msg = last_error();
    let c = std::ffi::CString::new(msg).unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
    LAST_ERR_C.with(|slot| {
        *slot.borrow_mut() = Some(c);
        slot.borrow().as_ref().unwrap().as_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_kind(r: *const OpaqueRtw) -> i32 {
    if r.is_null() {
        return -1;
    }
    unsafe { (*r).kind as i32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_bytes(r: *const OpaqueRtw, out_buf: *mut u8, out_cap: usize, out_len: *mut usize) -> i32 {
    ffi_guard(|| {
        if r.is_null() {
            return err(RTORCH_API_E_PARAM, "rtorch_api: null rtw");
        }
        let r = unsafe { &*r };
        if out_len.is_null() {
            return err(RTORCH_API_E_PARAM, "rtorch_api: null out_len");
        }
        let bytes = &r.bytes;
        unsafe { *out_len = bytes.len() };
        if bytes.len() > out_cap {
            // Report the required size in *out_len and fail; the caller reallocs.
            return err(RTORCH_API_E_PARAM, "rtorch_api: rtw bytes buffer too small");
        }
        if !out_buf.is_null() && !bytes.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, bytes.len());
            }
        }
        RTORCH_API_OK
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_decode(bytes: *const u8, len: usize) -> *mut OpaqueRtw {
    ffi_guard_ptr(|| {
        if bytes.is_null() && len > 0 {
            err(RTORCH_API_E_PARAM, "rtorch_api: null rtw bytes");
            return std::ptr::null_mut();
        }
        let buf = if len == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(bytes, len) } };
        match rtorch::rtw::decode(buf) {
            Ok(rtw) => {
                let boxed = Box::new(OpaqueRtw { kind: rtw.kind, bytes: rtw.data, manifest: rtw.manifest });
                Box::into_raw(boxed)
            }
            Err(e) => {
                err(RTORCH_API_E_PARSE, &format!("rtorch_api: rtw decode failed: {e}"));
                std::ptr::null_mut()
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_free(r: *mut OpaqueRtw) {
    if !r.is_null() {
        unsafe { drop(Box::from_raw(r)) };
    }
}

// ---- RTW Manifest accessors (RTorch 0.1.4) ----
// The RTW can carry a self-describing Manifest ("which library / where / format
// version"). These expose it so a cross-language consumer sees the same "what is
// this RTW" answers RTorch does. Strings are borrowed through a thread-local
// buffer valid until the next call on the same thread (same pattern as *_name).

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_has_manifest(r: *const OpaqueRtw) -> i32 {
    if r.is_null() {
        return 0;
    }
    if unsafe { &*r }.manifest.is_some() {
        1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_manifest_artifact_id(r: *const OpaqueRtw) -> *const std::ffi::c_char {
    if r.is_null() {
        return std::ptr::null();
    }
    match &unsafe { &*r }.manifest {
        Some(m) => cstr_return(&m.artifact_id),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_manifest_location(r: *const OpaqueRtw) -> *const std::ffi::c_char {
    if r.is_null() {
        return std::ptr::null();
    }
    match &unsafe { &*r }.manifest {
        Some(m) => cstr_return(&m.location),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_manifest_format_version(r: *const OpaqueRtw) -> *const std::ffi::c_char {
    if r.is_null() {
        return std::ptr::null();
    }
    match &unsafe { &*r }.manifest {
        Some(m) => cstr_return(&m.format_version),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_manifest_requires_count(r: *const OpaqueRtw) -> usize {
    if r.is_null() {
        return 0;
    }
    match &unsafe { &*r }.manifest {
        Some(m) => m.requires.len(),
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rtw_manifest_requires_dim(r: *const OpaqueRtw, i: usize) -> *const std::ffi::c_char {
    if r.is_null() {
        return std::ptr::null();
    }
    match &unsafe { &*r }.manifest {
        Some(m) => match m.requires.get(i) {
            Some(s) => cstr_return(s),
            None => std::ptr::null(),
        },
        None => std::ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// Model / memory payload accessors (M3). These wrap the raw model/memory payload
// bytes (the `data` field of an RTW of kind=KIND_MODEL / KIND_MEMORY) using
// rtorch's `rtw::decode_model` / `rtw::decode_memory`, and expose the fields over
// the C ABI so a cross-language consumer can inspect a model or memory without
// re-parsing the RTW container.
// ---------------------------------------------------------------------------

// A thread-local, reusable CString buffer for returning borrowed `const char*`.
// Names are owned `String`s inside the OpaqueModel; we cache their NUL-terminated
// form here (replaced on each call) so we can hand out a stable pointer without
// leaking (same pattern as `rtorch_api_last_error`).
fn cstr_return(s: &str) -> *const std::ffi::c_char {
    use std::cell::RefCell;
    thread_local! {
        static CSTR_BUF: RefCell<Option<std::ffi::CString>> = const { RefCell::new(None) };
    }
    let c = std::ffi::CString::new(s).unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
    CSTR_BUF.with(|slot| {
        *slot.borrow_mut() = Some(c);
        slot.borrow().as_ref().unwrap().as_ptr()
    })
}

// Sanity guard before delegating a Model/Memory payload to Rust/rtorch's decoder:
// the leading u32 count tells the decoder how many named-params / fragments to
// reserve. If that count is grossly larger than the byte length could possibly
// encode (each entry needs several bytes), the payload is malformed and rtorch's
// `Vec::with_capacity(count)` would attempt an absurd allocation and abort the
// process. We reject it here with a decode error instead of crashing.
//
// `min_entry` = a conservative lower bound of bytes per entry.
fn reject_oversized_count(buf: &[u8], min_entry: usize, what: &str) -> bool {
    if buf.len() < 4 {
        return false; // too short to even hold the count; let decoder report a clean error
    }
    let count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    // A count claiming more entries than the whole payload could hold (with at
    // least `min_entry` bytes each) is impossible -> reject before allocating.
    if count.checked_mul(min_entry).is_none_or(|needed| needed > buf.len()) {
        err(RTORCH_API_E_PARSE, &format!("rtorch_api: {what} payload count out of range"));
        true
    } else {
        false
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_decode(payload: *const u8, len: usize) -> *mut OpaqueModel {
    ffi_guard_ptr(|| {
        if payload.is_null() && len > 0 {
            err(RTORCH_API_E_PARAM, "rtorch_api: null model payload");
            return std::ptr::null_mut();
        }
        let buf = if len == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(payload, len) } };
        match rtorch::rtw::decode_model(buf) {
            Ok(model) => Box::into_raw(Box::new(OpaqueModel { model })),
            Err(e) => {
                err(RTORCH_API_E_PARSE, &format!("rtorch_api: model decode failed: {e}"));
                std::ptr::null_mut()
            }
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_free(m: *mut OpaqueModel) {
    if !m.is_null() {
        unsafe { drop(Box::from_raw(m)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_name(m: *const OpaqueModel) -> *const std::ffi::c_char {
    if m.is_null() {
        return std::ptr::null();
    }
    cstr_return(&unsafe { &*m }.model.name)
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_version(m: *const OpaqueModel) -> u32 {
    if m.is_null() {
        return 0;
    }
    unsafe { &*m }.model.version
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_num_params(m: *const OpaqueModel) -> usize {
    if m.is_null() {
        return 0;
    }
    unsafe { &*m }.model.params.len()
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_param_name(m: *const OpaqueModel, i: usize) -> *const std::ffi::c_char {
    if m.is_null() {
        return std::ptr::null();
    }
    let model = unsafe { &*m };
    match model.model.params.get(i) {
        Some(p) => cstr_return(&p.name),
        None => std::ptr::null(),
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_param_data(m: *const OpaqueModel, i: usize, out_numel: *mut usize) -> *const f32 {
    if m.is_null() || out_numel.is_null() {
        return std::ptr::null();
    }
    let model = unsafe { &*m };
    match model.model.params.get(i) {
        Some(p) => {
            unsafe { *out_numel = p.data.len() };
            p.data.as_ptr()
        }
        None => {
            unsafe { *out_numel = 0 };
            std::ptr::null()
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_param_shape_len(m: *const OpaqueModel, i: usize) -> usize {
    if m.is_null() {
        return 0;
    }
    let model = unsafe { &*m };
    model.model.params.get(i).map(|p| p.shape.len()).unwrap_or(0)
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_model_param_shape_dim(m: *const OpaqueModel, i: usize, j: usize) -> usize {
    if m.is_null() {
        return 0;
    }
    let model = unsafe { &*m };
    model
        .model
        .params
        .get(i)
        .and_then(|p| p.shape.get(j).copied())
        .map(|d| d as usize)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_memory_decode(payload: *const u8, len: usize) -> *mut OpaqueMemory {
    ffi_guard_ptr(|| {
        if payload.is_null() && len > 0 {
            err(RTORCH_API_E_PARAM, "rtorch_api: null memory payload");
            return std::ptr::null_mut();
        }
        let buf = if len == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(payload, len) } };
        // memory: each fragment >= 8B id + 4B state-len + min 1 float + 4B strength
        // (a v1 MemoryFragment is at least ~13 bytes; use a conservative 8).
        if reject_oversized_count(buf, 8, "memory") {
            return std::ptr::null_mut();
        }
        match rtorch::rtw::decode_memory(buf) {
            Ok(memory) => Box::into_raw(Box::new(OpaqueMemory { memory })),
            Err(e) => {
                err(RTORCH_API_E_PARSE, &format!("rtorch_api: memory decode failed: {e}"));
                std::ptr::null_mut()
            }
        }
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_memory_free(m: *mut OpaqueMemory) {
    if !m.is_null() {
        unsafe { drop(Box::from_raw(m)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_memory_num_frags(m: *const OpaqueMemory) -> usize {
    if m.is_null() {
        return 0;
    }
    unsafe { &*m }.memory.fragments.len()
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_memory_frag_id(m: *const OpaqueMemory, i: usize) -> u64 {
    if m.is_null() {
        return 0;
    }
    let mem = unsafe { &*m };
    mem.memory.fragments.get(i).map(|f| f.id).unwrap_or(0)
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_memory_frag_state(m: *const OpaqueMemory, i: usize, out_numel: *mut usize) -> *const f32 {
    if m.is_null() || out_numel.is_null() {
        return std::ptr::null();
    }
    let mem = unsafe { &*m };
    match mem.memory.fragments.get(i) {
        Some(f) => {
            unsafe { *out_numel = f.state.len() };
            f.state.as_ptr()
        }
        None => {
            unsafe { *out_numel = 0 };
            std::ptr::null()
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_memory_frag_strength(m: *const OpaqueMemory, i: usize) -> f32 {
    if m.is_null() {
        return 0.0;
    }
    let mem = unsafe { &*m };
    mem.memory.fragments.get(i).map(|f| f.strength).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Rule system (C ABI).
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_register_rule(
    name: *const std::ffi::c_char,
    cf: crate::rule::RuleCFn,
    userdata: *mut std::ffi::c_void,
) -> i32 {
    ffi_guard(|| {
        if name.is_null() {
            return err(RTORCH_API_E_PARAM, "rtorch_api: null rule name");
        }
        let n = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().to_string();
        crate::rule::register_c(&n, cf, userdata)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_rule_exists(name: *const std::ffi::c_char) -> i32 {
    ffi_guard(|| {
        if name.is_null() {
            return 0;
        }
        let n = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().to_string();
        if crate::rule::exists(&n) {
            1
        } else {
            0
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rtorch_api_run_rule(
    name: *const std::ffi::c_char,
    in_blobs: *const Blob,
    n_in: usize,
    out: *mut Blob,
    userdata: *mut std::ffi::c_void,
) -> i32 {
    ffi_guard(|| {
        if name.is_null() || out.is_null() {
            return err(RTORCH_API_E_PARAM, "rtorch_api: null name/out");
        }
        let n = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().to_string();
        let blobs = if n_in == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(in_blobs, n_in) }.to_vec()
        };
        let out = unsafe { &mut *out };
        crate::rule::run(&n, &blobs, out, userdata)
    })
}

// Silence unused-import warnings for helpers used only by tests.
#[allow(unused)]
fn _unused() {
    let _ = HashMap::<usize, usize>::new;
}
