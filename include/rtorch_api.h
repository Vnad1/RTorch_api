// RTorch_api — stable C ABI over the RTorch compute layer.
//
// This is the single bridge that Rust/C++/Python/C# all call. It sits ON TOP of
// RTorch (which stays compute-only); RTorch_api provides a richer, extensible
// interface plus a plug-in rule system for the wider community.
//
// Conventions (all four languages rely on these):
//   * `extern "C"`, no exceptions. Errors propagate via an `int rc` return +
//     `rtorch_api_last_error()`.
//   * Memory ownership is explicit: every `*_new`/`*_decode` returns an owned
//     handle you must release with the matching `*_free`.
//   * The Rust shell does all bounds/pointer checks and NEVER lets a Rust panic
//     cross the C boundary (returns rc != 0 instead).
#ifndef RTORCH_API_H
#define RTORCH_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Error codes.
// ---------------------------------------------------------------------------
enum {
    RTORCH_API_OK = 0,
    RTORCH_API_E_PARAM = 1,   // bad / null argument
    RTORCH_API_E_NOMEM = 2,   // allocation failed
    RTORCH_API_E_RT = 3,      // RTorch compute error
    RTORCH_API_E_IO = 4,      // file / I-O error
    RTORCH_API_E_PARSE = 5,   // bad RTW / payload
};

// Last error message (thread-local). Valid until the next call.
const char* rtorch_api_last_error(void);

// ---------------------------------------------------------------------------
// Tensor (Rust-safe vector view).
// ---------------------------------------------------------------------------
typedef struct rtorch_api_tensor rtorch_api_tensor;

// dtype — only RTORCH_API_DTYPE_F32 (0) is currently backed by the safe shell.
// F64/I32 are reserved; rtorch_api_tensor_new returns RTORCH_API_E_PARAM for them
// rather than silently storing an f32 buffer under another dtype (which would let
// a caller read the wrong element width).
enum {
    RTORCH_API_DTYPE_F32 = 0,
    RTORCH_API_DTYPE_F64 = 1, // reserved, not yet backed
    RTORCH_API_DTYPE_I32 = 2, // reserved, not yet backed
};

// Create a tensor of the given shape (row-major, f32). `dims` len = `rank`.
// Returns an owned handle, or NULL on error (check last_error).
rtorch_api_tensor* rtorch_api_tensor_new(const size_t* dims, size_t rank, int dtype);
void               rtorch_api_tensor_free(rtorch_api_tensor* t);

// Read-only access to the raw data (numel f32 values). Borrowed, not owned.
const float*       rtorch_api_tensor_data(const rtorch_api_tensor* t);
size_t             rtorch_api_tensor_numel(const rtorch_api_tensor* t);
size_t             rtorch_api_tensor_rank(const rtorch_api_tensor* t);
size_t             rtorch_api_tensor_dim(const rtorch_api_tensor* t, size_t i);
int                rtorch_api_tensor_dtype(const rtorch_api_tensor* t);

// ---------------------------------------------------------------------------
// Formula session — compile once / execute many (fixes "recompile every run").
// ---------------------------------------------------------------------------
typedef struct rtorch_api_session rtorch_api_session;

// Compile `formula_src` (.cpp path or .dll) with `ref_dir` include dir; returns an
// owned session, or NULL on error. Reuses RTorch's formula pipeline via the lib API.
rtorch_api_session* rtorch_api_session_compile(const char* formula_src, const char* ref_dir);
void                rtorch_api_session_free(rtorch_api_session* s);

// Run the compiled formula once with `n` input blobs. `out` buffer is provided by
// the caller (len = capacity); on return out.len is the bytes actually written,
// or rc != 0. `device` 0 = CPU, 1 = GPU.
typedef struct { const void* data; size_t len; } rtorch_api_blob;
int rtorch_api_session_execute(rtorch_api_session* s,
                               const rtorch_api_blob* in, size_t n_in,
                               rtorch_api_blob* out, int device);

// ---------------------------------------------------------------------------
// RTW container (result / kernel / model / memory).
// ---------------------------------------------------------------------------
typedef struct rtorch_api_rtw rtorch_api_rtw;

int  rtorch_api_rtw_kind(const rtorch_api_rtw* r);          // KIND_* from rtw.rs
// Copy the RTW's payload bytes into `out_buf` (capacity = out_cap). `out_len`
// receives the number of bytes written on success; returns RTORCH_API_OK or
// RTORCH_API_E_PARAM if the buffer is too small. No ownership transfer — the
// caller owns `out_buf`.
int  rtorch_api_rtw_bytes(const rtorch_api_rtw* r, uint8_t* out_buf, size_t out_cap, size_t* out_len);
rtorch_api_rtw* rtorch_api_rtw_decode(const uint8_t* bytes, size_t len);
void            rtorch_api_rtw_free(rtorch_api_rtw* r);

// RTW self-describing Manifest (RTorch 0.1.4): "which library / where / format
// version / required capabilities". `*_artifact_id`/`*_location`/
// `*_format_version`/`*_requires_dim` return a borrowed `const char*` valid until
// the next RTorch_api call on the same thread; NULL if the RTW has no Manifest.
int         rtorch_api_rtw_has_manifest(const rtorch_api_rtw* r);
const char* rtorch_api_rtw_manifest_artifact_id(const rtorch_api_rtw* r);
const char* rtorch_api_rtw_manifest_location(const rtorch_api_rtw* r);
const char* rtorch_api_rtw_manifest_format_version(const rtorch_api_rtw* r);
size_t      rtorch_api_rtw_manifest_requires_count(const rtorch_api_rtw* r);
const char* rtorch_api_rtw_manifest_requires_dim(const rtorch_api_rtw* r, size_t i);

// ---------------------------------------------------------------------------
// Model / memory payload accessors.
// ---------------------------------------------------------------------------
typedef struct rtorch_api_model rtorch_api_model;
typedef struct rtorch_api_memory rtorch_api_memory;

rtorch_api_model*  rtorch_api_model_decode(const uint8_t* payload, size_t len);
void               rtorch_api_model_free(rtorch_api_model* m);
const char*        rtorch_api_model_name(const rtorch_api_model* m);
uint32_t           rtorch_api_model_version(const rtorch_api_model* m);
size_t             rtorch_api_model_num_params(const rtorch_api_model* m);
const char*        rtorch_api_model_param_name(const rtorch_api_model* m, size_t i);
const float*       rtorch_api_model_param_data(const rtorch_api_model* m, size_t i, size_t* out_numel);
size_t             rtorch_api_model_param_shape_len(const rtorch_api_model* m, size_t i);
size_t             rtorch_api_model_param_shape_dim(const rtorch_api_model* m, size_t i, size_t j);

rtorch_api_memory* rtorch_api_memory_decode(const uint8_t* payload, size_t len);
void               rtorch_api_memory_free(rtorch_api_memory* m);
size_t             rtorch_api_memory_num_frags(const rtorch_api_memory* m);
uint64_t           rtorch_api_memory_frag_id(const rtorch_api_memory* m, size_t i);
const float*       rtorch_api_memory_frag_state(const rtorch_api_memory* m, size_t i, size_t* out_numel);
float              rtorch_api_memory_frag_strength(const rtorch_api_memory* m, size_t i);

// ---------------------------------------------------------------------------
// Rule system — plug-in rules added by users (community extensibility).
// ---------------------------------------------------------------------------
typedef void (*rtorch_api_rule_fn)(const rtorch_api_blob* in, size_t n_in,
                                   rtorch_api_blob* out, void* userdata);

// Register a rule `name` -> `fn`. Returns RTORCH_API_OK or error (dup name, etc).
int  rtorch_api_register_rule(const char* name, rtorch_api_rule_fn fn, void* userdata);
int  rtorch_api_rule_exists(const char* name);
// Run a registered rule over `n` input blobs, writing into `out`.
int  rtorch_api_run_rule(const char* name, const rtorch_api_blob* in, size_t n_in,
                         rtorch_api_blob* out, void* userdata);

#ifdef __cplusplus
}
#endif

#endif // RTORCH_API_H
