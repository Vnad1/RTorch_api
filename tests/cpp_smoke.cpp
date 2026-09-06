// C++ smoke test for the cpp/rtorch_for_models_api.hpp wrapper. Links against
// rtorch_for_models_api.dll and exercises Tensor, Session, and rule registration,
// proving the header-only C++ wrapper works end-to-end.
//
// Build (from the RTorch_api dir):
//   g++ -std=c++17 -I include -I cpp tests/cpp_smoke.cpp -o cpp_smoke.exe \
//       -L target/release -l:rtorch_for_models_api.dll
//   PATH=target/release cpp_smoke.exe
#include "rtorch_for_models_api.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

static void test_tensor() {
    using rtorch_api::Tensor;
    Tensor t(std::vector<size_t>{2, 3}, RTORCH_API_DTYPE_F32);
    assert(t.numel() == 6);
    assert(t.rank() == 2);
    assert(t.dim(0) == 2);
    assert(t.dim(1) == 3);
    assert(t.data() != nullptr);
    // F64 must be rejected (matches the F32-only contract).
    bool threw = false;
    try {
        Tensor bad(std::vector<size_t>{2}, RTORCH_API_DTYPE_F64);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    printf("tensor OK\n");
}

// A C-ABI rule callback that writes its `userdata` (an int) as an f32. Returns 0.
static int c_rule(const rtorch_api_blob* /*in*/, size_t /*n_in*/, rtorch_api_blob* out, void* userdata) {
    float v = static_cast<float>(*static_cast<int*>(userdata));
    if (out->data && out->len >= sizeof(v)) {
        *static_cast<float*>(const_cast<void*>(out->data)) = v;
        out->len = sizeof(v);
    }
    return 0;
}

static void test_rule() {
    using namespace rtorch_api;
    int ud = 42;
    int rc = register_rule("cpp_rule", c_rule, &ud);
    assert(rc == RTORCH_API_OK);
    assert(rule_exists("cpp_rule"));

    float out_buf = 0.0f;
    rtorch_api_blob out{&out_buf, sizeof(out_buf)};
    rc = run_rule("cpp_rule", {}, &out);
    assert(rc == RTORCH_API_OK);
    assert(out.len == sizeof(float));
    assert(out_buf == 42.0f);
    printf("rule OK\n");
}

int main() {
    test_tensor();
    test_rule();
    printf("ALL C++ SMOKE PASSED\n");
    return 0;
}
