# RTorch_for_models_api

**RTorch_for_models_api** is a cross-language plugin layer over **RTorch** — a
general-purpose compute framework. RTorch itself stays compute-only; this plugin
sits on top and exposes a **stable C ABI** (`include/rtorch_api.h`) plus bindings
for Rust, C++, Python, and C#, so a broader community can use RTorch's computation
without talking to Rust directly, and can register plug-in **rules**.

> RTorch (compute layer) is the foundation. Striker (a model framework) builds on
> RTorch — it is a separate project, not part of this one. This repo is the API /
> extension layer for the community.

## Versioning

RTorch_for_models_api uses **Semantic Versioning**. The current version is **`0.0.1`**.

- `MAJOR.MINOR.PATCH`; `0.0.1` is the first development release.
- PATCH = bug-fix / compatible change; MINOR = backward-compatible feature;
  MAJOR = breaking change.

The RTorch *compute layer* (the project this plugin sits on top of) uses its own
**date-based** release number `YYYY.MM.DD.N` (e.g. `2026.09.06.1`) — the two are
separate and not tied to each other.

## What it exposes

Through one stable C ABI (`include/rtorch_api.h`), every function is `extern "C"`,
returns an `int rc` (0 = `RTORCH_API_OK`), and reports failures through the
thread-local `rtorch_api_last_error()`. Handles are explicit: every `*_new` /
`*_decode` is paired with a `*_free`.

- **Tensor** — F32 tensor safe view (`new / free / data / numel / rank / dim / dtype`).
- **Formula session** — compile-once / execute-many (`compile / execute / free`),
  no per-call recompile of a `.cpp`.
- **RTW container** — self-contained artifact (`kind / bytes / decode / free`,
  plus the RTW self-describing **Manifest**: `has_manifest / manifest_artifact_id /
  manifest_location / manifest_format_version / manifest_requires_count /
  manifest_requires_dim`).
- **Model / Memory** — payload accessors (`decode / name / version / num_params /
  param_*`; `num_frags / frag_id / frag_state / frag_strength`).
- **Rules** — community plug-ins (`register_rule / rule_exists / run_rule`), added
  by writing code (a C function pointer + optional userdata).

## Language bindings

| Lang | Path | Role |
|---|---|---|
| Rust (safe shell) | `src/` (lib) | memory-safety vector, the single C ABI implementation |
| C++ | `cpp/rtorch_for_models_api.hpp` | header-only RAII wrapper (performance hot-path) |
| Python | `python/rtorch_for_models_api.py` | ctypes glue, Python-first ergonomics |
| C# | `csharp/RTorchForModelsApi.cs` | P/Invoke, rich managed API |

## Build

```sh
cargo build --release
```

Produces `target/release/rtorch_for_models_api.dll` (cdylib) + the rlib.

## Run the smoke tests

### Rust
```sh
cargo test          # 10 tests: tensor/session/rtw/model/memory/rule
```

### C++ (needs g++, links the built DLL)
```sh
g++ -std=c++17 -I include -I cpp tests/cpp_smoke.cpp \
    target/release/rtorch_for_models_api.dll -o tests/cpp_smoke.exe
PATH="target/release;$PATH" tests/cpp_smoke.exe
```

### Python (needs Python 3 + the DLL on PATH)
```sh
PATH="target/release;$PATH" python tests/python_smoke.py
```

### C# (needs .NET SDK)
```sh
dotnet build csharp/RtApiSmoke/RtApiSmoke.csproj -c Release
```

## Notes

- **dtype**: only `RTORCH_API_DTYPE_F32` (0) is currently backed by the safe shell;
  F64/I32 are rejected (`E_PARAM`) rather than silently stored as f32 (which would
  let a caller read the wrong element width).
- Only the C ABI's declared surface is stable. RTorch core is never modified.

## License

RTorch_for_models_api is licensed under the **GNU Lesser General Public License,
version 3.0 (LGPL-3.0)** — see `LICENSE`. It is a plug-in that links against
RTorch (also LGPL-3.0) through its public API; the LGPL allows it to be used and
relinked as a library.
