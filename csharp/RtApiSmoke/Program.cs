using System;
using System.Linq;
using System.Text;
using RTorchApi;

class Program
{
    static void Main()
    {
        TestTensor();
        TestRule();
        Console.WriteLine("ALL C# SMOKE PASSED");
    }

    static void TestTensor()
    {
        using (var t = new RtorchApi.Tensor(new ulong[] { 2, 3 }, RtorchApi.DTYPE_F32))
        {
            AssertEq(6UL, t.Numel, "numel");
            AssertEq(2UL, t.Rank, "rank");
            AssertEq(2UL, t.Dim(0), "dim0");
            AssertEq(3UL, t.Dim(1), "dim1");
            var shape = t.Shape;
            AssertEq(2UL, shape[0], "shape0");
            AssertEq(3UL, shape[1], "shape1");
            AssertEq(6, t.Data.Length, "data len");
        }
        // F64 must be rejected.
        try
        {
            using var bad = new RtorchApi.Tensor(new ulong[] { 2 }, RtorchApi.DTYPE_F64);
            throw new Exception("F64 should have been rejected");
        }
        catch (RTorchApiException) { }
        Console.WriteLine("C# tensor OK");
    }

    static void TestRule()
    {
        // Register a C# rule fn that writes a fixed float into the out blob.
        _ruleFn = (inPtr, nIn, outPtr, userdata) =>
        {
            float v = 42.0f;
            if (outPtr != IntPtr.Zero)
            {
                Blob blob = System.Runtime.InteropServices.Marshal.PtrToStructure<Blob>(outPtr);
                if (blob.Data != IntPtr.Zero && blob.Len >= 4)
                {
                    byte[] bytes = BitConverter.GetBytes(v);
                    System.Runtime.InteropServices.Marshal.Copy(bytes, 0, blob.Data, bytes.Length);
                    blob = new Blob(blob.Data, (ulong)bytes.Length);
                    System.Runtime.InteropServices.Marshal.StructureToPtr(blob, outPtr, false);
                }
            }
            return RtorchApi.OK;
        };
        int rc = RtorchApi.RegisterRule("cs_rule", _ruleFn, IntPtr.Zero);
        AssertEq(RtorchApi.OK, rc, "register_rule rc");
        AssertEq(true, RtorchApi.RuleExists("cs_rule"), "rule_exists");
        rc = RtorchApi.RunRule("cs_rule", Array.Empty<byte[]>(), out byte[] res, IntPtr.Zero);
        AssertEq(RtorchApi.OK, rc, "run_rule rc");
        float val = BitConverter.ToSingle(res, 0);
        AssertEq(42.0f, val, "rule output");
        Console.WriteLine("C# rule OK");
    }

    static void AssertEq<T>(T expected, T actual, string what)
    {
        if (!object.Equals(expected, actual))
            throw new Exception($"{what}: expected {expected} got {actual}");
    }

    static RtorchApi.RuleFn _ruleFn;
}
