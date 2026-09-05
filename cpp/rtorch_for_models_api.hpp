// RTorch_for_models_api — C++ thin wrapper (performance hot-path) over the RTorch
// stable C ABI (`include/rtorch_api.h`).
//
// This is header-only and zero-cost where it can be: it owns every opaque handle
// with RAII so a C++ caller never manually frees, and it turns the C ABI's `int rc`
// returns into C++ exceptions (std::runtime_error) on failure. For latency-critical
// paths you can still call the C ABI directly; the classes here exist for clean
// ownership + ergonomics.
//
// Build: link against rtorch_for_models_api.dll (the cdylib), or drop this into a
// project that already has the C ABI available.
#ifndef RTORCH_FOR_MODELS_API_CPP_HPP
#define RTORCH_FOR_MODELS_API_CPP_HPP

#include "rtorch_api.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtorch_api {

// Throw a std::runtime_error carrying the thread-local C ABI error message.
[[noreturn]] inline void throw_last_error(const char* what) {
    const char* msg = rtorch_api_last_error();
    std::string detail = (msg && *msg) ? msg : "unknown RTorch_api error";
    throw std::runtime_error(std::string(what) + ": " + detail);
}

// RAII Tensor — owns an rtorch_api_tensor*, frees on destruction.
class Tensor {
public:
    // Create an F32 tensor of `dims`. Throws on invalid dtype/args.
    explicit Tensor(const std::vector<size_t>& dims, int dtype = RTORCH_API_DTYPE_F32) {
        t_ = rtorch_api_tensor_new(dims.empty() ? nullptr : dims.data(), dims.size(), dtype);
        if (!t_) throw_last_error("rtorch_api::Tensor::Tensor");
    }
    // Adopt an existing handle (ownership transfers).
    explicit Tensor(rtorch_api_tensor* t) : t_(t) {}
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept : t_(o.t_) { o.t_ = nullptr; }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) { reset(); std::swap(t_, o.t_); }
        return *this;
    }
    ~Tensor() { reset(); }

    void reset() {
        if (t_) { rtorch_api_tensor_free(t_); t_ = nullptr; }
    }

    size_t numel() const { return rtorch_api_tensor_numel(t_); }
    size_t rank() const { return rtorch_api_tensor_rank(t_); }
    size_t dim(size_t i) const { return rtorch_api_tensor_dim(t_, i); }
    int dtype() const { return rtorch_api_tensor_dtype(t_); }
    const float* data() const { return rtorch_api_tensor_data(t_); }
    const float* operator*() const { return data(); }
    rtorch_api_tensor* get() const { return t_; }

private:
    rtorch_api_tensor* t_ = nullptr;
};

// RAII Session — a compiled formula, executable many times.
class Session {
public:
    // Compile `formula_src` (.cpp path or .dll). Throws on failure.
    explicit Session(const std::string& formula_src, const std::string& ref_dir = "") {
        s_ = rtorch_api_session_compile(formula_src.c_str(), ref_dir.empty() ? nullptr : ref_dir.c_str());
        if (!s_) throw_last_error("rtorch_api::Session::Session");
    }
    // Adopt an existing session handle.
    explicit Session(rtorch_api_session* s) : s_(s) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& o) noexcept : s_(o.s_) { o.s_ = nullptr; }
    Session& operator=(Session&& o) noexcept { if (this != &o) { release(); std::swap(s_, o.s_); } return *this; }
    ~Session() { release(); }

    // Run the formula once. `inputs` are the raw byte blobs; `device` 0 = CPU, 1 = GPU.
    // Returns the produced output bytes. Throws on C ABI error.
    std::vector<uint8_t> run(const std::vector<std::vector<uint8_t>>& inputs, int device) const {
        std::vector<rtorch_api_blob> in_blobs;
        in_blobs.reserve(inputs.size());
        for (const auto& b : inputs) {
            in_blobs.push_back(rtorch_api_blob{b.empty() ? nullptr : b.data(), b.size()});
        }
        // First call asks output size: pass a NULL out buffer with 0 cap; the
        // session fills out.len with actual size (returns E_PARAM if > cap=0, but
        // reports the required size). Then allocate and run again.
        // Simpler robust approach: probe size by giving a 0-capacity buffer, then
        // allocate enough and run.
        std::vector<uint8_t> out;
        size_t want = 0;
        {
            rtorch_api_blob out_blob{nullptr, 0};
            int rc = rtorch_api_session_execute(s_, in_blobs.data(), in_blobs.size(), &out_blob, device);
            want = out_blob.len; // the session reports required bytes even on E_PARAM
            if (rc != 0 && rc != RTORCH_API_E_PARAM) {
                // compiler/parse error, not a size probe issue
                throw_last_error("rtorch_api::Session::run(probe)");
            }
        }
        out.resize(want);
        rtorch_api_blob out_blob{out.empty() ? nullptr : out.data(), out.size()};
        int rc = rtorch_api_session_execute(s_, in_blobs.data(), in_blobs.size(), &out_blob, device);
        if (rc != 0) throw_last_error("rtorch_api::Session::run");
        out.resize(out_blob.len);
        return out;
    }

    rtorch_api_session* get() const { return s_; }

private:
    void release() { if (s_) { rtorch_api_session_free(s_); s_ = nullptr; } }
    rtorch_api_session* s_ = nullptr;
};

// RAII RTW — owns an rtorch_api_rtw*.
class Rtw {
public:
    explicit Rtw(const std::vector<uint8_t>& bytes) {
        r_ = rtorch_api_rtw_decode(bytes.empty() ? nullptr : bytes.data(), bytes.size());
        if (!r_) throw_last_error("rtorch_api::Rtw::Rtw");
    }
    explicit Rtw(rtorch_api_rtw* r) : r_(r) {}
    ~Rtw() { if (r_) rtorch_api_rtw_free(r_); }
    Rtw(const Rtw&) = delete;
    Rtw& operator=(const Rtw&) = delete;

    int kind() const { return rtorch_api_rtw_kind(r_); }
    // Copy the payload into a caller-owned buffer; returns the byte count.
    std::vector<uint8_t> bytes() const {
        size_t len = 0;
        // Probe.
        rtorch_api_rtw_bytes(r_, nullptr, 0, &len);
        std::vector<uint8_t> out(len);
        int rc = rtorch_api_rtw_bytes(r_, out.empty() ? nullptr : out.data(), out.size(), &len);
        if (rc != 0) throw_last_error("rtorch_api::Rtw::bytes");
        out.resize(len);
        return out;
    }
    rtorch_api_rtw* get() const { return r_; }

    // RTW self-describing Manifest (RTorch 0.1.4). Empty when the RTW has none.
    bool has_manifest() const { return rtorch_api_rtw_has_manifest(r_) != 0; }
    std::string artifact_id() const {
        const char* s = rtorch_api_rtw_manifest_artifact_id(r_);
        return s ? s : "";
    }
    std::string location() const {
        const char* s = rtorch_api_rtw_manifest_location(r_);
        return s ? s : "";
    }
    std::string format_version() const {
        const char* s = rtorch_api_rtw_manifest_format_version(r_);
        return s ? s : "";
    }
    size_t requires_count() const { return rtorch_api_rtw_manifest_requires_count(r_); }
    std::string requires_dim(size_t i) const {
        const char* s = rtorch_api_rtw_manifest_requires_dim(r_, i);
        return s ? s : "";
    }

private:
    rtorch_api_rtw* r_ = nullptr;
};

// RAII Model — owns an rtorch_api_model*.
class Model {
public:
    explicit Model(const std::vector<uint8_t>& payload) {
        m_ = rtorch_api_model_decode(payload.empty() ? nullptr : payload.data(), payload.size());
        if (!m_) throw_last_error("rtorch_api::Model::Model");
    }
    explicit Model(rtorch_api_model* m) : m_(m) {}
    ~Model() { if (m_) rtorch_api_model_free(m_); }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    std::string name() const { const char* s = rtorch_api_model_name(m_); return s ? s : ""; }
    uint32_t version() const { return rtorch_api_model_version(m_); }
    size_t num_params() const { return rtorch_api_model_num_params(m_); }
    std::string param_name(size_t i) const {
        const char* s = rtorch_api_model_param_name(m_, i);
        return s ? s : "";
    }
    std::vector<float> param_data(size_t i) const {
        size_t n = 0;
        const float* d = rtorch_api_model_param_data(m_, i, &n);
        return (d && n) ? std::vector<float>(d, d + n) : std::vector<float>{};
    }
    std::vector<size_t> param_shape(size_t i) const {
        std::vector<size_t> s;
        size_t r = rtorch_api_model_param_shape_len(m_, i);
        s.reserve(r);
        for (size_t j = 0; j < r; ++j) s.push_back(rtorch_api_model_param_shape_dim(m_, i, j));
        return s;
    }
    rtorch_api_model* get() const { return m_; }

private:
    rtorch_api_model* m_ = nullptr;
};

// RAII Memory — owns an rtorch_api_memory*.
class Memory {
public:
    explicit Memory(const std::vector<uint8_t>& payload) {
        m_ = rtorch_api_memory_decode(payload.empty() ? nullptr : payload.data(), payload.size());
        if (!m_) throw_last_error("rtorch_api::Memory::Memory");
    }
    explicit Memory(rtorch_api_memory* m) : m_(m) {}
    ~Memory() { if (m_) rtorch_api_memory_free(m_); }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    size_t num_frags() const { return rtorch_api_memory_num_frags(m_); }
    uint64_t frag_id(size_t i) const { return rtorch_api_memory_frag_id(m_, i); }
    std::vector<float> frag_state(size_t i) const {
        size_t n = 0;
        const float* d = rtorch_api_memory_frag_state(m_, i, &n);
        return (d && n) ? std::vector<float>(d, d + n) : std::vector<float>{};
    }
    float frag_strength(size_t i) const { return rtorch_api_memory_frag_strength(m_, i); }
    rtorch_api_memory* get() const { return m_; }

private:
    rtorch_api_memory* m_ = nullptr;
};

// Rule registration helper. `fn` may be a C function pointer (matching
// rtorch_api_rule_fn) with an optional `userdata` — the registered `userdata` is
// passed on the rule's invocations.
inline int register_rule(const std::string& name, rtorch_api_rule_fn fn, void* userdata = nullptr) {
    return rtorch_api_register_rule(name.c_str(), fn, userdata);
}
inline bool rule_exists(const std::string& name) {
    return rtorch_api_rule_exists(name.c_str()) == 1;
}
inline int run_rule(const std::string& name, const std::vector<rtorch_api_blob>& in, rtorch_api_blob* out, void* userdata = nullptr) {
    return rtorch_api_run_rule(name.c_str(), in.empty() ? nullptr : in.data(), in.size(), out, userdata);
}

// Convenience: the last error message as a std::string.
inline std::string last_error() {
    const char* s = rtorch_api_last_error();
    return s ? s : "";
}

} // namespace rtorch_api

#endif // RTORCH_FOR_MODELS_API_CPP_HPP
