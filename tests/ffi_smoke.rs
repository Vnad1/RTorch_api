//! C-ABI smoke test for RTorch_api M1.
//!
//! Exercises the stable C ABI entry points the same way a C++/Python/C# caller
//! would: tensor construction, a formula session compile+execute, and the rule
//! registry. These call the `extern "C"` functions directly from Rust (same ABI),
//! which proves the exported symbols behave.

use rtorch_for_models_api::ffi::*;

#[test]
fn tensor_roundtrip() {
    // [2, 3] f32 tensor -> numel 6, dims correct
    let dims = [2usize, 3usize];
    let t = rtorch_api_tensor_new(dims.as_ptr(), 2, 0);
    assert!(!t.is_null(), "tensor_new returned null");
    assert_eq!(rtorch_api_tensor_numel(t), 6);
    assert_eq!(rtorch_api_tensor_rank(t), 2);
    assert_eq!(rtorch_api_tensor_dim(t, 0), 2);
    assert_eq!(rtorch_api_tensor_dim(t, 1), 3);
    assert_eq!(rtorch_api_tensor_dtype(t), 0);
    // data pointer is non-null and 6 f32s readable.
    let data = rtorch_api_tensor_data(t);
    assert!(!data.is_null());
    let slice = unsafe { std::slice::from_raw_parts(data, 6) };
    assert_eq!(slice.len(), 6);
    rtorch_api_tensor_free(t);
}

#[test]
fn tensor_dtype_contract() {
    // F64 (1) / I32 (2) must be rejected, NOT silently stored as f32 (that would
    // let a caller read the wrong element width = memory bug). This locks in the
    // F32-only contract.
    let dims = [2usize, 3usize];
    assert!(rtorch_api_tensor_new(dims.as_ptr(), 2, 1).is_null(), "F64 should be rejected");
    assert!(rtorch_api_last_error_str().contains("F32"), "error should mention F32");
    assert!(rtorch_api_tensor_new(dims.as_ptr(), 2, 2).is_null(), "I32 should be rejected");
    // null dims with rank>0 rejected
    assert!(rtorch_api_tensor_new(std::ptr::null(), 2, 0).is_null(), "null dims should be rejected");
}

#[test]
fn session_execute_null_guard() {
    // Null session / null out -> E_PARAM (no panic, no UB).
    let rc = rtorch_api_session_execute(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null_mut(), 0);
    assert_eq!(rc, 1, "null session should be E_PARAM, got {rc}");
}

#[test]
fn rule_not_found() {
    // Running a never-registered rule -> E_PARAM (rule not found), no panic.
    let name = b"no_such_rule\0";
    let mut out_bytes = [0u8; 16];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 16 };
    let rc = rtorch_api_run_rule(name.as_ptr() as *const _, std::ptr::null(), 0, &mut out_blob, std::ptr::null_mut());
    assert_eq!(rc, 1, "missing rule should be E_PARAM, got {rc}");
}

#[test]
fn session_compile_execute() {
    // Point at a real rtorch formula so formula::run executes. Use the delta.dll
    // from the rtorch example output or compile the trig formula source.
    // We compile the trig formula .cpp (needs g++; skip if unavailable).
    if !gpp_available() {
        return;
    }
    let formula = concat!(env!("CARGO_MANIFEST_DIR"), "/../rtorch/examples/formula_trig.cpp");
    if !std::path::Path::new(formula).exists() {
        return;
    }
    // The C ABI takes a NUL-terminated path; wrap in CString.
    let formula_c = std::ffi::CString::new(formula).unwrap();
    let s = rtorch_api_session_compile(formula_c.as_ptr(), std::ptr::null());
    assert!(!s.is_null(), "session_compile returned null");

    // 1 angle (0.0) -> 3 floats out (sin, cos, tan).
    let in_val = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
    let in_blob = Blob { data: in_val.as_ptr() as *const _, len: 16 };
    let mut out_bytes = [0u8; 4096];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 4096 };
    let rc = rtorch_api_session_execute(s, &in_blob, 1, &mut out_blob, 0);
    assert_eq!(rc, 0, "session_execute rc={rc} ({})", rtorch_api_last_error_str());
    // 1 angle, but trig expects floatelem count; len=16 => 4 angles => 12 floats out.
    assert_eq!(out_blob.len, 12 * 4, "expected 48 bytes, got {}", out_blob.len);
    // sin(0)=0, cos(0)=1, tan(0)=0 at first triple.
    let o = unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, 12) };
    assert!(o[0].abs() < 1e-6, "sin(0)={}", o[0]);
    assert!((o[1] - 1.0).abs() < 1e-6, "cos(0)={}", o[1]);
    assert!(o[2].abs() < 1e-6, "tan(0)={}", o[2]);

    rtorch_api_session_free(s);
}

#[test]
fn rule_register_c_abi() {
    // Demonstrate the C-ABI fn-pointer path (M2): register an `extern "C"` rule
    // fn + userdata via rtorch_api_register_rule, then run it the way a C++
    // caller would. userdata is a plain i32 we pass-through.
    let name = b"m2_c_rule\0";
    let mut userdata: i32 = 7;
    let rc = rtorch_api_register_rule(
        name.as_ptr() as *const _,
        Some(c_rule),
        std::ptr::addr_of_mut!(userdata).cast(),
    );
    assert_eq!(rc, 0, "register_c rc={rc} ({})", rtorch_api_last_error_str());
    assert_eq!(rtorch_api_rule_exists(name.as_ptr() as *const _), 1);

    // run -> the C fn reads its userdata (7), writes it as an f32.
    let mut out_bytes = [0u8; 16];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 16 };
    let rc2 = rtorch_api_run_rule(name.as_ptr() as *const _, std::ptr::null(), 0, &mut out_blob, std::ptr::null_mut());
    assert_eq!(rc2, 0, "run_c rc={rc2}");
    let val = f32::from_le_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
    assert_eq!(val, 7.0, "rule_c output = {val} (expected 7.0 from userdata)");
    assert_eq!(out_blob.len, 4);
}

#[test]
fn rule_c_error_propagates() {
    // P0-1: a C rule callback that returns a non-zero rc must be surfaced by
    // run_rule (the rc is no longer swallowed as success).
    let name = b"m2_c_err\0";
    let rc = rtorch_api_register_rule(name.as_ptr() as *const _, Some(c_rule_err), std::ptr::null_mut());
    assert_eq!(rc, 0, "register rc={rc}");
    let mut out_bytes = [0u8; 16];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 16 };
    let rc2 = rtorch_api_run_rule(name.as_ptr() as *const _, std::ptr::null(), 0, &mut out_blob, std::ptr::null_mut());
    assert_eq!(rc2, 7, "callback error rc must propagate, got {rc2}");
}

#[test]
fn run_rule_null_blobs_with_count_rejected() {
    // P0-2: in_blobs == NULL with n_in > 0 must be rejected (E_PARAM), not
    // dereferenced (which would be UB).
    let name = b"any\0";
    let mut out_bytes = [0u8; 16];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 16 };
    let rc = rtorch_api_run_rule(name.as_ptr() as *const _, std::ptr::null(), 100, &mut out_blob, std::ptr::null_mut());
    assert_eq!(rc, 1, "null in_blobs with n_in>0 must be E_PARAM, got {rc}");
}

// The `extern "C"` rule callback, matching rtorch_api_rule_fn in the header.
// Returns 0 = ok (so the rule's rc propagates out of run_rule).
unsafe extern "C" fn c_rule(
    _in: *const Blob,
    _n_in: usize,
    out: *mut Blob,
    userdata: *mut std::ffi::c_void,
) -> i32 {
    let ud = unsafe { &*(userdata.cast::<i32>()) };
    let val = [*ud as f32];
    unsafe {
        std::ptr::copy_nonoverlapping(val.as_ptr() as *const u8, (*out).data as *mut u8, 4);
        (*out).len = 4;
    }
    0
}

// A rule that always returns a non-zero error code, so we can verify the rc
// propagates back to run_rule (P0-1: C rule errors must not be swallowed).
unsafe extern "C" fn c_rule_err(
    _in: *const Blob,
    _n_in: usize,
    _out: *mut Blob,
    _userdata: *mut std::ffi::c_void,
) -> i32 {
    7 // arbitrary non-zero error
}

#[test]
fn rule_register_and_run() {
    // Register a rule that computes a static value into the output blob.
    let name = b"m1_rule\0";
    // We wrap a Rust closure as RuleFn; register via the Rust API (C ABI fn-ptr
    // wiring is a M2 refinement, see note).
    let f: rtorch_for_models_api::rule::RuleFn = Box::new(|_in, out, _ud| {
        let v = [42.0f32]; // one f32
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out.data as *mut u8, 4);
        }
        out.len = 4;
        rtorch_for_models_api::err::RTORCH_API_OK
    });
    let rc = rtorch_for_models_api::rule::register(name_of(name), f);
    assert_eq!(rc, 0, "register rc={rc}");
    assert!(rtorch_api_rule_exists(name.as_ptr() as *const _) == 1);

    let mut out_bytes = [0u8; 16];
    let mut out_blob = Blob { data: out_bytes.as_mut_ptr() as *const _, len: 16 };
    let rc2 = rtorch_api_run_rule(name.as_ptr() as *const _, std::ptr::null(), 0, &mut out_blob, std::ptr::null_mut());
    assert_eq!(rc2, 0, "run rule rc={rc2}");
    let val = f32::from_le_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
    assert!((val - 42.0).abs() < 1e-6, "rule output = {val}");
}

fn name_of(bytes: &[u8]) -> &str {
    // name is NUL-terminated; strip the NUL
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("")
}

fn rtorch_api_last_error_str() -> String {
    let p = rtorch_api_last_error();
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().to_string() }
}

#[test]
fn model_accessors() {
    // Build a real Model via RTorch, encode to the model payload bytes, then
    // decode through the C ABI and read every accessor the way a C++/Python caller
    // would. This proves the accessors return the actual parsed fields.
    let model = rtorch::rtw::Model {
        name: "toy_nn".to_string(),
        version: 7,
        params: vec![
            rtorch::rtw::NamedTensor {
                name: "w".to_string(),
                shape: vec![2, 3],
                dtype: 0,
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
            rtorch::rtw::NamedTensor {
                name: "b".to_string(),
                shape: vec![3],
                dtype: 0,
                data: vec![0.5, -0.5, 2.0],
            },
        ],
        opt: None,
    };
    let payload = rtorch::rtw::encode_model(&model);

    let m = rtorch_api_model_decode(payload.as_ptr(), payload.len());
    assert!(!m.is_null(), "model_decode null ({})", rtorch_api_last_error_str());

    assert!(!rtorch_api_model_name(m).is_null(), "name null");
    unsafe {
        let name = std::ffi::CStr::from_ptr(rtorch_api_model_name(m)).to_string_lossy();
        assert_eq!(name, "toy_nn");
    }
    assert_eq!(rtorch_api_model_version(m), 7);
    assert_eq!(rtorch_api_model_num_params(m), 2);

    // param 0: w
    assert_eq!(unsafe_cstr(rtorch_api_model_param_name(m, 0)), "w");
    assert_eq!(rtorch_api_model_param_shape_len(m, 0), 2);
    assert_eq!(rtorch_api_model_param_shape_dim(m, 0, 0), 2);
    assert_eq!(rtorch_api_model_param_shape_dim(m, 0, 1), 3);
    let mut numel = 0usize;
    let data = rtorch_api_model_param_data(m, 0, &mut numel);
    let slice = unsafe { std::slice::from_raw_parts(data, numel) };
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // param 1: b
    assert_eq!(unsafe_cstr(rtorch_api_model_param_name(m, 1)), "b");
    assert_eq!(rtorch_api_model_param_shape_dim(m, 1, 0), 3);

    // out-of-range -> 0/null (no panic, no UB)
    assert!(rtorch_api_model_param_data(m, 9, &mut numel).is_null());

    rtorch_api_model_free(m);
}

#[test]
fn memory_accessors() {
    let memory = rtorch::rtw::Memory {
        fragments: vec![
            rtorch::rtw::MemoryFragment { id: 11, state: vec![0.1, 0.2, 0.3], strength: 0.9 },
            rtorch::rtw::MemoryFragment { id: 22, state: vec![0.4, 0.5], strength: 0.4 },
        ],
    };
    let payload = rtorch::rtw::encode_memory(&memory);

    let m = rtorch_api_memory_decode(payload.as_ptr(), payload.len());
    assert!(!m.is_null(), "memory_decode null ({})", rtorch_api_last_error_str());

    assert_eq!(rtorch_api_memory_num_frags(m), 2);
    assert_eq!(rtorch_api_memory_frag_id(m, 0), 11);
    assert_eq!(rtorch_api_memory_frag_id(m, 1), 22);
    assert_eq!(rtorch_api_memory_frag_strength(m, 0), 0.9);
    assert_eq!(rtorch_api_memory_frag_strength(m, 1), 0.4);

    let mut numel = 0usize;
    let st = rtorch_api_memory_frag_state(m, 0, &mut numel);
    let slice = unsafe { std::slice::from_raw_parts(st, numel) };
    assert_eq!(slice, &[0.1, 0.2, 0.3]);

    // out-of-range -> 0/null
    assert!(rtorch_api_memory_frag_id(m, 99) == 0);
    assert!(rtorch_api_memory_frag_state(m, 99, &mut numel).is_null());

    rtorch_api_memory_free(m);
}

fn unsafe_cstr(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().to_string() }
}

#[test]
fn rtw_container_roundtrip() {
    // Build an RTW result container via RTorch, decode through the C ABI, and
    // read kind + payload bytes back (the "RTW 完整读写" path). `rtorch_api_rtw_bytes`
    // returns the `data` payload exactly as `rtw::decode` captured it.
    let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let rtw = rtorch::rtw::Rtw {
        kind: rtorch::rtw::KIND_RESULT,
        dtype: rtorch::rtw::DTYPE_FP32,
        shape: vec![2, 3],
        data: data.clone(),
        kernel: None,
        manifest: None,
    };
    let bytes = rtorch::rtw::encode(&rtw);

    let r = rtorch_api_rtw_decode(bytes.as_ptr(), bytes.len());
    assert!(!r.is_null(), "rtw_decode null ({})", rtorch_api_last_error_str());
    assert_eq!(rtorch_api_rtw_kind(r), rtorch::rtw::KIND_RESULT as i32);

    let mut out_len = 0usize;
    let mut buf = vec![0u8; data.len()];
    let rc = rtorch_api_rtw_bytes(r, buf.as_mut_ptr(), buf.len(), &mut out_len);
    assert_eq!(rc, 0, "rtw_bytes rc={rc} ({})", rtorch_api_last_error_str());
    assert_eq!(out_len, data.len());
    assert_eq!(&buf[..], &data[..]);

    // Buffer-too-small path: report size, return E_PARAM, no overrun.
    let mut small = vec![0u8; 2];
    let mut small_len = 0usize;
    let rc2 = rtorch_api_rtw_bytes(r, small.as_mut_ptr(), small.len(), &mut small_len);
    assert_eq!(rc2, 1, "expected E_PARAM for small buffer (got {rc2})");
    assert_eq!(small_len, data.len(), "should report required size");

    rtorch_api_rtw_free(r);
}

#[test]
fn rtw_manifest_accessors() {
    // Build an RTW carrying a self-describing Manifest, decode through the C ABI,
    // and read the "which library / where / format version / requires" answers.
    let data: Vec<u8> = vec![1.0f32, 2.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let rtw = rtorch::rtw::Rtw {
        kind: rtorch::rtw::KIND_RESULT,
        dtype: rtorch::rtw::DTYPE_FP32,
        shape: vec![2],
        data: data.clone(),
        kernel: None,
        manifest: Some(rtorch::rtw::Manifest {
            artifact_id: "com.example.model".to_string(),
            location: "/models/my.rtw".to_string(),
            format_version: rtorch::rtw::RTW_FORMAT_VERSION.to_string(),
            requires: vec!["compute.session".to_string(), "memory.fragment".to_string()],
        }),
    };
    let bytes = rtorch::rtw::encode(&rtw);
    let r = rtorch_api_rtw_decode(bytes.as_ptr(), bytes.len());
    assert!(!r.is_null(), "rtw_decode null ({})", rtorch_api_last_error_str());

    assert_eq!(rtorch_api_rtw_has_manifest(r), 1);
    assert_eq!(unsafe_cstr(rtorch_api_rtw_manifest_artifact_id(r)), "com.example.model");
    assert_eq!(unsafe_cstr(rtorch_api_rtw_manifest_location(r)), "/models/my.rtw");
    assert_eq!(unsafe_cstr(rtorch_api_rtw_manifest_format_version(r)), rtorch::rtw::RTW_FORMAT_VERSION);
    assert_eq!(rtorch_api_rtw_manifest_requires_count(r), 2);
    assert_eq!(unsafe_cstr(rtorch_api_rtw_manifest_requires_dim(r, 0)), "compute.session");
    assert_eq!(unsafe_cstr(rtorch_api_rtw_manifest_requires_dim(r, 1)), "memory.fragment");
    // Out-of-range -> null.
    assert!(rtorch_api_rtw_manifest_requires_dim(r, 99).is_null());

    rtorch_api_rtw_free(r);
}

#[test]
fn rtw_manifest_boundaries() {
    // No-manifest RTW: accessors return 0/null without panic.
    let data = vec![0u8; 4];
    let rtw = rtorch::rtw::Rtw {
        kind: rtorch::rtw::KIND_RESULT,
        dtype: rtorch::rtw::DTYPE_FP32,
        shape: vec![1],
        data,
        kernel: None,
        manifest: None,
    };
    let bytes = rtorch::rtw::encode(&rtw);
    let r = rtorch_api_rtw_decode(bytes.as_ptr(), bytes.len());
    assert!(!r.is_null());
    assert_eq!(rtorch_api_rtw_has_manifest(r), 0);
    assert!(rtorch_api_rtw_manifest_artifact_id(r).is_null());
    assert!(rtorch_api_rtw_manifest_location(r).is_null());
    assert!(rtorch_api_rtw_manifest_format_version(r).is_null());
    assert_eq!(rtorch_api_rtw_manifest_requires_count(r), 0);
    assert!(rtorch_api_rtw_manifest_requires_dim(r, 0).is_null());
    rtorch_api_rtw_free(r);

    // Null handle: all accessors return 0/null, no panic.
    assert_eq!(rtorch_api_rtw_has_manifest(std::ptr::null()), 0);
    assert!(rtorch_api_rtw_manifest_artifact_id(std::ptr::null()).is_null());
    assert!(rtorch_api_rtw_manifest_requires_dim(std::ptr::null(), 0).is_null());
}

fn gpp_available() -> bool {
    if std::env::var_os("RTORCH_GXX").is_some() {
        return true;
    }
    std::process::Command::new("where")
        .arg("g++")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
