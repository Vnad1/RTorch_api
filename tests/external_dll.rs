//! External-loader test: load the built `rtorch_for_models_api.dll` as a *foreign*
//! library (via LoadLibrary/GetProcAddress, i.e. exactly what Python's ctypes and
//! C#'s P/Invoke do), resolve the C ABI symbols by name, and call them. This proves
//! the DLL is truly loadable and callable from outside the crate — the strongest
//! ABI/UB check that doesn't depend on the Rust code being co-linked.
//!
//! Run: `cargo test --test external_dll` (debug build loads target/debug DLL).

use std::ffi::c_void;
use std::sync::OnceLock;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(h: *mut c_void) -> i32;
}

type TensorNewFn = unsafe extern "C" fn(*const usize, usize, i32) -> *mut c_void;
type TensorFreeFn = unsafe extern "C" fn(*mut c_void);
type TensorNumelFn = unsafe extern "C" fn(*const c_void) -> usize;
type TensorDataFn = unsafe extern "C" fn(*const c_void) -> *const f32;

struct Dll {
    handle: *mut c_void,
}
unsafe impl Send for Dll {}
unsafe impl Sync for Dll {}

impl Drop for Dll {
    fn drop(&mut self) {
        unsafe { FreeLibrary(self.handle) };
    }
}

fn dll() -> &'static OnceLock<Dll> {
    static DLL: OnceLock<Dll> = OnceLock::new();
    &DLL
}

#[cfg(windows)]
fn resolve<T>(handle: *mut c_void, name: &str) -> Option<T> {
    let cname = std::ffi::CString::new(name).ok()?;
    let p = unsafe { GetProcAddress(handle, cname.as_ptr() as *const u8) };
    if p.is_null() {
        return None;
    }
    if std::mem::size_of::<T>() != std::mem::size_of::<usize>() {
        return None;
    }
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

#[cfg(windows)]
fn load() -> *mut c_void {
    use std::os::windows::ffi::OsStrExt;
    // Prefer the freshly-built DLL next to this test (target/debug), then release.
    let cand = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/target/debug/rtorch_for_models_api.dll"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/target/release/rtorch_for_models_api.dll"),
    ];
    for p in &cand {
        let w: Vec<u16> = std::path::Path::new(p).as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let h = unsafe { LoadLibraryW(w.as_ptr()) };
        if !h.is_null() {
            return h;
        }
    }
    std::ptr::null_mut()
}

#[cfg(windows)]
fn handle() -> *mut c_void {
    let h = dll().get_or_init(|| Dll { handle: load() }).handle;
    assert!(!h.is_null(), "failed to load rtorch_for_models_api.dll");
    h
}

#[test]
#[cfg(windows)]
fn external_load_and_call() {
    let h = handle();
    let new_fn: TensorNewFn = resolve(h, "rtorch_api_tensor_new").expect("tensor_new symbol");
    let free_fn: TensorFreeFn = resolve(h, "rtorch_api_tensor_free").expect("tensor_free symbol");
    let numel_fn: TensorNumelFn = resolve(h, "rtorch_api_tensor_numel").expect("tensor_numel symbol");
    let data_fn: TensorDataFn = resolve(h, "rtorch_api_tensor_data").expect("tensor_data symbol");

    // [2,3] f32 -> numel 6
    let dims = [2usize, 3usize];
    let t = unsafe { new_fn(dims.as_ptr(), 2, 0) };
    assert!(!t.is_null(), "tensor_new returned null");
    assert_eq!(unsafe { numel_fn(t) }, 6, "numel mismatch");
    let data = unsafe { data_fn(t) };
    assert!(!data.is_null(), "data null");
    let slice = unsafe { std::slice::from_raw_parts(data, 6) };
    assert_eq!(slice.len(), 6);
    unsafe { free_fn(t) };
}

#[test]
#[cfg(windows)]
fn external_bad_name_returns_none() {
    // A symbol that isn't there -> resolve() returns None (no UB).
    let h = handle();
    assert!(resolve::<TensorNumelFn>(h, "rtorch_api_does_not_exist").is_none());
}
