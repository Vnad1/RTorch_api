"""RTorch_for_models_api — Python binding (glue) over the RTorch stable C ABI.

Loads ``rtorch_for_models_api.dll`` (the cdylib next to this package, or the path
in the ``RTORCH_API_DLL`` env var) via ctypes and exposes a small, ergonomic,
Python-first wrapper: every opaque handle is owned by a Python object (freed on
``Close()`` / ``__del__``), and any C ABI error (non-zero rc) raises
:class:`RTorchApiError` with the thread-local ``rtorch_api_last_error()`` message.

Only the C ABI's F32 tensor dtype is backed; ``RTorchApiError`` is raised for
F64/I32 (matching the library's contract).
"""

from __future__ import annotations

import ctypes as _ct
import os as _os
from typing import Iterable, Optional, Sequence as _Seq


# ---------------------------------------------------------------------------
# Loading the DLL.
# ---------------------------------------------------------------------------

def _find_dll() -> str:
    env = _os.environ.get("RTORCH_API_DLL")
    if env:
        return env
    # Typical layouts: package dir, repo root, repo target/release.
    here = _os.path.dirname(_os.path.abspath(__file__))
    candidates = [
        here.replace("\\python", ""),           # repo root (this dir's parent-ish)
        here,                                    # this dir
        _os.path.join(here, "target", "release"),
        _os.path.join(_os.path.dirname(here), "target", "release"),
    ]
    for d in candidates:
        for name in ("rtorch_for_models_api.dll", "rtorch_for_models_api.so", "librtorch_for_models_api.so"):
            p = _os.path.join(d, name)
            if _os.path.exists(p):
                return p
    # Last resort: rely on OS loader search (add target/release to PATH).
    return "rtorch_for_models_api.dll"


_dll = _ct.CDLL(_find_dll())

# ---------------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------------
RTORCH_API_OK = 0
RTORCH_API_E_PARAM = 1
RTORCH_API_E_NOMEM = 2
RTORCH_API_E_RT = 3
RTORCH_API_E_IO = 4
RTORCH_API_E_PARSE = 5

# dtype (only F32 is backed; F64/I32 raise RTorchApiError).
RTORCH_API_DTYPE_F32 = 0
RTORCH_API_DTYPE_F64 = 1
RTORCH_API_DTYPE_I32 = 2

_dll.rtorch_api_last_error.restype = _ct.c_char_p


def last_error() -> Optional[str]:
    p = _dll.rtorch_api_last_error()
    return p.decode("utf-8", "replace") if p else None


class RTorchApiError(RuntimeError):
    """Raised when the C ABI returns a non-zero rc."""

    def __init__(self, what: str, rc: int):
        self.rc = rc
        super().__init__(f"{what}: rc={rc} ({last_error() or 'unknown'})")


def _check(rc: int, what: str) -> int:
    if rc != RTORCH_API_OK:
        raise RTorchApiError(what, rc)
    return rc


# ---------------------------------------------------------------------------
# C ABI signatures.
# ---------------------------------------------------------------------------
class _Blob(_ct.Structure):
    _fields_ = [("data", _ct.c_void_p), ("len", _ct.c_size_t)]


_dll.rtorch_api_tensor_new.restype = _ct.c_void_p
_dll.rtorch_api_tensor_new.argtypes = [_ct.POINTER(_ct.c_size_t), _ct.c_size_t, _ct.c_int]
_dll.rtorch_api_tensor_free.argtypes = [_ct.c_void_p]
_dll.rtorch_api_tensor_data.restype = _ct.POINTER(_ct.c_float)
_dll.rtorch_api_tensor_data.argtypes = [_ct.c_void_p]
_dll.rtorch_api_tensor_numel.restype = _ct.c_size_t
_dll.rtorch_api_tensor_numel.argtypes = [_ct.c_void_p]
_dll.rtorch_api_tensor_rank.restype = _ct.c_size_t
_dll.rtorch_api_tensor_rank.argtypes = [_ct.c_void_p]
_dll.rtorch_api_tensor_dim.restype = _ct.c_size_t
_dll.rtorch_api_tensor_dim.argtypes = [_ct.c_void_p, _ct.c_size_t]
_dll.rtorch_api_tensor_dtype.restype = _ct.c_int
_dll.rtorch_api_tensor_dtype.argtypes = [_ct.c_void_p]

_dll.rtorch_api_session_compile.restype = _ct.c_void_p
_dll.rtorch_api_session_compile.argtypes = [_ct.c_char_p, _ct.c_char_p]
_dll.rtorch_api_session_free.argtypes = [_ct.c_void_p]
_dll.rtorch_api_session_execute.restype = _ct.c_int
_dll.rtorch_api_session_execute.argtypes = [_ct.c_void_p, _ct.POINTER(_Blob), _ct.c_size_t, _ct.POINTER(_Blob), _ct.c_int]

_dll.rtorch_api_rtw_kind.restype = _ct.c_int
_dll.rtorch_api_rtw_kind.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_bytes.restype = _ct.c_int
_dll.rtorch_api_rtw_bytes.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_uint8), _ct.c_size_t, _ct.POINTER(_ct.c_size_t)]
_dll.rtorch_api_rtw_decode.restype = _ct.c_void_p
_dll.rtorch_api_rtw_decode.argtypes = [_ct.POINTER(_ct.c_uint8), _ct.c_size_t]
_dll.rtorch_api_rtw_free.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_has_manifest.restype = _ct.c_int
_dll.rtorch_api_rtw_has_manifest.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_manifest_artifact_id.restype = _ct.c_char_p
_dll.rtorch_api_rtw_manifest_artifact_id.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_manifest_location.restype = _ct.c_char_p
_dll.rtorch_api_rtw_manifest_location.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_manifest_format_version.restype = _ct.c_char_p
_dll.rtorch_api_rtw_manifest_format_version.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_manifest_requires_count.restype = _ct.c_size_t
_dll.rtorch_api_rtw_manifest_requires_count.argtypes = [_ct.c_void_p]
_dll.rtorch_api_rtw_manifest_requires_dim.restype = _ct.c_char_p
_dll.rtorch_api_rtw_manifest_requires_dim.argtypes = [_ct.c_void_p, _ct.c_size_t]

_dll.rtorch_api_model_decode.restype = _ct.c_void_p
_dll.rtorch_api_model_decode.argtypes = [_ct.POINTER(_ct.c_uint8), _ct.c_size_t]
_dll.rtorch_api_model_free.argtypes = [_ct.c_void_p]
_dll.rtorch_api_model_name.restype = _ct.c_char_p
_dll.rtorch_api_model_name.argtypes = [_ct.c_void_p]
_dll.rtorch_api_model_version.restype = _ct.c_uint32
_dll.rtorch_api_model_version.argtypes = [_ct.c_void_p]
_dll.rtorch_api_model_num_params.restype = _ct.c_size_t
_dll.rtorch_api_model_num_params.argtypes = [_ct.c_void_p]
_dll.rtorch_api_model_param_name.restype = _ct.c_char_p
_dll.rtorch_api_model_param_name.argtypes = [_ct.c_void_p, _ct.c_size_t]
_dll.rtorch_api_model_param_data.restype = _ct.POINTER(_ct.c_float)
_dll.rtorch_api_model_param_data.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.POINTER(_ct.c_size_t)]
_dll.rtorch_api_model_param_shape_len.restype = _ct.c_size_t
_dll.rtorch_api_model_param_shape_len.argtypes = [_ct.c_void_p, _ct.c_size_t]
_dll.rtorch_api_model_param_shape_dim.restype = _ct.c_size_t
_dll.rtorch_api_model_param_shape_dim.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.c_size_t]

_dll.rtorch_api_memory_decode.restype = _ct.c_void_p
_dll.rtorch_api_memory_decode.argtypes = [_ct.POINTER(_ct.c_uint8), _ct.c_size_t]
_dll.rtorch_api_memory_free.argtypes = [_ct.c_void_p]
_dll.rtorch_api_memory_num_frags.restype = _ct.c_size_t
_dll.rtorch_api_memory_num_frags.argtypes = [_ct.c_void_p]
_dll.rtorch_api_memory_frag_id.restype = _ct.c_uint64
_dll.rtorch_api_memory_frag_id.argtypes = [_ct.c_void_p, _ct.c_size_t]
_dll.rtorch_api_memory_frag_state.restype = _ct.POINTER(_ct.c_float)
_dll.rtorch_api_memory_frag_state.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.POINTER(_ct.c_size_t)]
_dll.rtorch_api_memory_frag_strength.restype = _ct.c_float
_dll.rtorch_api_memory_frag_strength.argtypes = [_ct.c_void_p, _ct.c_size_t]

_dll.rtorch_api_register_rule.restype = _ct.c_int
_dll.rtorch_api_register_rule.argtypes = [_ct.c_char_p, _ct.c_void_p, _ct.c_void_p]
_dll.rtorch_api_rule_exists.restype = _ct.c_int
_dll.rtorch_api_rule_exists.argtypes = [_ct.c_char_p]
_dll.rtorch_api_run_rule.restype = _ct.c_int
_dll.rtorch_api_run_rule.argtypes = [_ct.c_char_p, _ct.POINTER(_Blob), _ct.c_size_t, _ct.POINTER(_Blob), _ct.c_void_p]


# ---------------------------------------------------------------------------
# Python wrappers.
# ---------------------------------------------------------------------------
class Tensor:
    def __init__(self, dims: Iterable[int], dtype: int = 0):
        dims_l = list(dims)
        arr = (_ct.c_size_t * len(dims_l))(*dims_l) if dims_l else None
        self._h = _dll.rtorch_api_tensor_new(arr, len(dims_l), dtype)
        if not self._h:
            raise RTorchApiError("tensor_new", RTORCH_API_E_PARAM)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _dll.rtorch_api_tensor_free(self._h)
            self._h = None

    __del__ = close

    @property
    def numel(self) -> int:
        return _dll.rtorch_api_tensor_numel(self._h)

    @property
    def rank(self) -> int:
        return _dll.rtorch_api_tensor_rank(self._h)

    def dim(self, i: int) -> int:
        return _dll.rtorch_api_tensor_dim(self._h, i)

    @property
    def dtype(self) -> int:
        return _dll.rtorch_api_tensor_dtype(self._h)

    @property
    def data(self):
        """The f32 buffer as a ctypes float array (numel elements)."""
        p = _dll.rtorch_api_tensor_data(self._h)
        if not p:
            return None
        n = self.numel
        return _ct.cast(p, _ct.POINTER(_ct.c_float * n)).contents

    @property
    def shape(self) -> tuple:
        return tuple(self.dim(i) for i in range(self.rank))

    def numpy(self):
        """Return a copy as a list-of-lists (row-major) — no numpy dependency here."""
        vals = list(self.data)
        if not vals:
            return []
        shp = self.shape
        if len(shp) == 1:
            return vals
        # reshape row-major
        rows, cols = 1, 1
        for d in shp:
            rows *= d
        # build nested from shape
        return _reshape(vals, list(shp))


def _reshape(vals: list, shape: list) -> list:
    if len(shape) == 1:
        return vals
    # Each child holds the product of the *remaining* dims, not of all-with-n.
    child = 1
    for d in shape[1:]:
        child *= d
    return [_reshape(vals[i * child:(i + 1) * child], shape[1:]) for i in range(shape[0])]


class Session:
    def __init__(self, formula_src: str, ref_dir: str = ""):
        self._h = _dll.rtorch_api_session_compile(
            formula_src.encode(), ref_dir.encode() if ref_dir else None)
        if not self._h:
            raise RTorchApiError("session_compile", RTORCH_API_E_PARAM)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _dll.rtorch_api_session_free(self._h)
            self._h = None

    __del__ = close

    def run(self, inputs: Sequence[tuple | bytes | bytearray], device: int = 0) -> bytes:
        blobs = (_Blob * len(inputs))()
        for i, b in enumerate(inputs):
            if isinstance(b, tuple):
                data, length = b
                blobs[i].data = _ct.cast(data, _ct.c_void_p)
                blobs[i].len = length
            else:
                arr = (bytes(b) if isinstance(b, (bytearray, memoryview)) else b).__class__  # noqa: F841
                cbuf = _ct.create_string_buffer(bytes(b), len(b))
                blobs[i].data = _ct.cast(cbuf, _ct.c_void_p)
                blobs[i].len = len(b)
        # Probe output size (0-capacity buffer reports required length).
        probe = _Blob(None, 0)
        rc = _dll.rtorch_api_session_execute(self._h, blobs, len(inputs), _ct.byref(probe), device)
        if rc not in (0, RTORCH_API_E_PARAM):
            raise RTorchApiError("session_execute(probe)", rc)
        want = probe.len
        out = _ct.create_string_buffer(max(want, 1))
        out_blob = _Blob(_ct.cast(out, _ct.c_void_p), len(out))
        rc = _dll.rtorch_api_session_execute(self._h, blobs, len(inputs), _ct.byref(out_blob), device)
        if rc != RTORCH_API_OK:
            raise RTorchApiError("session_execute", rc)
        return out.raw[:out_blob.len]


class Rtw:
    def __init__(self, payload: bytes):
        buf = _ct.create_string_buffer(payload, len(payload)) if payload else None
        self._h = _dll.rtorch_api_rtw_decode(_ct.cast(buf, _ct.POINTER(_ct.c_uint8)) if buf else None, len(payload))
        if not self._h:
            raise RTorchApiError("rtw_decode", RTORCH_API_E_PARSE)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _dll.rtorch_api_rtw_free(self._h)
            self._h = None

    __del__ = close

    @property
    def kind(self) -> int:
        return _dll.rtorch_api_rtw_kind(self._h)

    def bytes(self) -> bytes:
        n = _ct.c_size_t(0)
        _dll.rtorch_api_rtw_bytes(self._h, None, 0, _ct.byref(n))
        buf = _ct.create_string_buffer(max(n.value, 1))
        rc = _dll.rtorch_api_rtw_bytes(self._h, _ct.cast(buf, _ct.POINTER(_ct.c_uint8)), len(buf), _ct.byref(n))
        if rc != RTORCH_API_OK:
            raise RTorchApiError("rtw_bytes", rc)
        return buf.raw[:n.value]

    @property
    def has_manifest(self) -> bool:
        return _dll.rtorch_api_rtw_has_manifest(self._h) != 0

    @property
    def artifact_id(self) -> str:
        p = _dll.rtorch_api_rtw_manifest_artifact_id(self._h)
        return p.decode() if p else ""

    @property
    def location(self) -> str:
        p = _dll.rtorch_api_rtw_manifest_location(self._h)
        return p.decode() if p else ""

    @property
    def format_version(self) -> str:
        p = _dll.rtorch_api_rtw_manifest_format_version(self._h)
        return p.decode() if p else ""

    @property
    def requires(self) -> list:
        n = _dll.rtorch_api_rtw_manifest_requires_count(self._h)
        out = []
        for i in range(n):
            p = _dll.rtorch_api_rtw_manifest_requires_dim(self._h, i)
            out.append(p.decode() if p else "")
        return out


class Model:
    def __init__(self, payload: bytes):
        buf = _ct.create_string_buffer(payload, len(payload)) if payload else None
        self._h = _dll.rtorch_api_model_decode(_ct.cast(buf, _ct.POINTER(_ct.c_uint8)) if buf else None, len(payload))
        if not self._h:
            raise RTorchApiError("model_decode", RTORCH_API_E_PARSE)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _dll.rtorch_api_model_free(self._h)
            self._h = None

    __del__ = close

    @property
    def name(self) -> str:
        return (_dll.rtorch_api_model_name(self._h) or b"").decode()

    @property
    def version(self) -> int:
        return _dll.rtorch_api_model_version(self._h)

    @property
    def num_params(self) -> int:
        return _dll.rtorch_api_model_num_params(self._h)

    def param_name(self, i: int) -> str:
        return (_dll.rtorch_api_model_param_name(self._h, i) or b"").decode()

    def param_data(self, i: int) -> list:
        n = _ct.c_size_t(0)
        p = _dll.rtorch_api_model_param_data(self._h, i, _ct.byref(n))
        if not p or n.value == 0:
            return []
        return [p[j] for j in range(n.value)]

    def param_shape(self, i: int) -> tuple:
        r = _dll.rtorch_api_model_param_shape_len(self._h, i)
        return tuple(_dll.rtorch_api_model_param_shape_dim(self._h, i, j) for j in range(r))


class Memory:
    def __init__(self, payload: bytes):
        buf = _ct.create_string_buffer(payload, len(payload)) if payload else None
        self._h = _dll.rtorch_api_memory_decode(_ct.cast(buf, _ct.POINTER(_ct.c_uint8)) if buf else None, len(payload))
        if not self._h:
            raise RTorchApiError("memory_decode", RTORCH_API_E_PARSE)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _dll.rtorch_api_memory_free(self._h)
            self._h = None

    __del__ = close

    @property
    def num_frags(self) -> int:
        return _dll.rtorch_api_memory_num_frags(self._h)

    def frag_id(self, i: int) -> int:
        return _dll.rtorch_api_memory_frag_id(self._h, i)

    def frag_state(self, i: int) -> list:
        n = _ct.c_size_t(0)
        p = _dll.rtorch_api_memory_frag_state(self._h, i, _ct.byref(n))
        if not p or n.value == 0:
            return []
        return [p[j] for j in range(n.value)]

    def frag_strength(self, i: int) -> float:
        return _dll.rtorch_api_memory_frag_strength(self._h, i)


# ---------------------------------------------------------------------------
# Rule system.
# ---------------------------------------------------------------------------
_RuleCFn = _ct.CFUNCTYPE(None, _ct.POINTER(_Blob), _ct.c_size_t, _ct.POINTER(_Blob), _ct.c_void_p)


def register_rule(name: str, fn, userdata=None) -> int:
    """Register `fn` (a Python callable taking (in_blobs, out_blob, userdata)) as
    the rule `name`. Returns rc. `userdata` is passed through on each invocation."""
    cf = _RuleCFn(fn)
    # keep a reference so the callback stays alive
    _rule_refs[name] = cf
    return _dll.rtorch_api_register_rule(name.encode(), _ct.cast(cf, _ct.c_void_p), userdata)


def rule_exists(name: str) -> bool:
    return _dll.rtorch_api_rule_exists(name.encode()) == 1


def run_rule(name: str, in_blobs: Sequence[tuple | bytes], userdata=None) -> bytearray:
    blobs = (_Blob * len(in_blobs))()
    for i, b in enumerate(in_blobs):
        if isinstance(b, tuple):
            data, length = b
            blobs[i].data = _ct.cast(data, _ct.c_void_p)
            blobs[i].len = length
        else:
            cbuf = _ct.create_string_buffer(bytes(b), len(b))
            blobs[i].data = _ct.cast(cbuf, _ct.c_void_p)
            blobs[i].len = len(b)
    out = _ct.create_string_buffer(4096)
    out_blob = _Blob(_ct.cast(out, _ct.c_void_p), len(out))
    rc = _dll.rtorch_api_run_rule(name.encode(), blobs, len(in_blobs), _ct.byref(out_blob), userdata)
    if rc != RTORCH_API_OK:
        raise RTorchApiError("run_rule", rc)
    return bytearray(out.raw[:out_blob.len])


_rule_refs: dict[str, "_RuleCFn"] = {}
