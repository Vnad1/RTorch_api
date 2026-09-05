// Compile-only check: prove include/rtorch_api.h parses as clean C and that every
// declared symbol has an identifiable type (so C/C++/Python/C# bindings work).
// Never linked; -c only.
#include "rtorch_api.h"
#include <stddef.h>

// Call every declared function in an unevaluated context so the compiler must
// resolve each declaration (catches typos / wrong names / wrong arg counts).
void use_all(void) {
    (void)rtorch_api_last_error;

    (void)rtorch_api_tensor_new;
    (void)rtorch_api_tensor_free;
    (void)rtorch_api_tensor_data;
    (void)rtorch_api_tensor_numel;
    (void)rtorch_api_tensor_rank;
    (void)rtorch_api_tensor_dim;
    (void)rtorch_api_tensor_dtype;

    (void)rtorch_api_session_compile;
    (void)rtorch_api_session_free;
    (void)rtorch_api_session_execute;

    (void)rtorch_api_rtw_kind;
    (void)rtorch_api_rtw_bytes;
    (void)rtorch_api_rtw_decode;
    (void)rtorch_api_rtw_free;

    (void)rtorch_api_model_decode;
    (void)rtorch_api_model_free;
    (void)rtorch_api_model_name;
    (void)rtorch_api_model_version;
    (void)rtorch_api_model_num_params;
    (void)rtorch_api_model_param_name;
    (void)rtorch_api_model_param_data;
    (void)rtorch_api_model_param_shape_len;
    (void)rtorch_api_model_param_shape_dim;

    (void)rtorch_api_memory_decode;
    (void)rtorch_api_memory_free;
    (void)rtorch_api_memory_num_frags;
    (void)rtorch_api_memory_frag_id;
    (void)rtorch_api_memory_frag_state;
    (void)rtorch_api_memory_frag_strength;

    (void)rtorch_api_register_rule;
    (void)rtorch_api_rule_exists;
    (void)rtorch_api_run_rule;
}
