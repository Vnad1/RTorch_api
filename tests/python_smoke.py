# Python smoke test for python/rtorch_for_models_api.py.
# Runs against the real rtorch_for_models_api.dll (PATH must include target/release).
import ctypes as ct
import os
import struct
import sys

# Make the binding importable from the repo tree.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import rtorch_for_models_api as ra  # noqa: E402


def test_tensor():
    t = ra.Tensor([2, 3])
    assert t.numel == 6
    assert t.rank == 2
    assert t.shape == (2, 3)
    assert t.dtype == ra.RTORCH_API_DTYPE_F32
    assert len(list(t.data)) == 6
    assert t.numpy() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    t.close()
    try:
        ra.Tensor([2], ra.RTORCH_API_DTYPE_F64)
        raise AssertionError("F64 should have been rejected")
    except ra.RTorchApiError:
        pass
    print("python tensor OK")


def test_rule():
    # The rule writes its register-time `userdata` (a float) into the out blob.
    ud = ct.pointer(ct.c_float(3.0))
    ud_addr = ct.cast(ud, ct.c_void_p).value

    def my_rule(in_ptr, n_in, out_blob, userdata):
        # userdata is the c_void_p we registered; read the float it points to.
        val = 7.0
        if userdata:
            val = ct.cast(userdata, ct.POINTER(ct.c_float)).contents.value
        ct.memmove(out_blob.contents.data, ct.byref(ct.c_float(val)), 4)
        out_blob.contents.len = 4
        return 0

    rc = ra.register_rule("py_rule", my_rule, ud_addr)
    assert rc == ra.RTORCH_API_OK
    assert ra.rule_exists("py_rule")
    res = ra.run_rule("py_rule", [])
    assert struct.unpack("<f", bytes(res[:4]))[0] == 3.0
    print("python rule OK")


if __name__ == "__main__":
    test_tensor()
    test_rule()
    print("ALL PYTHON SMOKE PASSED")
