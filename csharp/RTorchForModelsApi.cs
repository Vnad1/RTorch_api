// RTorch_for_models_api — C# P/Invoke binding over the RTorch stable C ABI.
//
// This is the "巨量引用/丰富高层 API" layer: a thin, managed, disposable wrapper
// around every C ABI entry point in `include/rtorch_api.h`. Memory ownership is
// explicit (every *_new/*_decode is paired with a *_free via IDisposable); errors
// surface as RTorchApiException carrying rc + the thread-local error message.
//
// The DLL is resolved by the runtime's native loader (search PATH, or set
// DllImport SearchPath / load it explicitly first). Place rtorch_for_models_api.dll
// on PATH or next to the app.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RTorchApi
{
    // -----------------------------------------------------------------------
    // Error codes (mirror include/rtorch_api.h).
    // -----------------------------------------------------------------------
    public static class RtorchApi
    {
        public const int OK = 0;
        public const int E_PARAM = 1;
        public const int E_NOMEM = 2;
        public const int E_RT = 3;
        public const int E_IO = 4;
        public const int E_PARSE = 5;

        public const int DTYPE_F32 = 0;
        public const int DTYPE_F64 = 1; // reserved, not backed
        public const int DTYPE_I32 = 2; // reserved, not backed

        private const string Dll = "rtorch_for_models_api";

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_last_error();

        // Tensor
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_tensor_new([In] ulong[] dims, ulong rank, int dtype);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rtorch_api_tensor_free(IntPtr t);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_tensor_data(IntPtr t);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_tensor_numel(IntPtr t);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_tensor_rank(IntPtr t);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_tensor_dim(IntPtr t, ulong i);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_tensor_dtype(IntPtr t);

        // Session
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_session_compile([MarshalAs(UnmanagedType.LPStr)] string formulaSrc, [MarshalAs(UnmanagedType.LPStr)] string refDir);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rtorch_api_session_free(IntPtr s);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_session_execute(IntPtr s, [In] Blob[] blobs, ulong nIn, [In, Out] Blob[] outBlob, int device);

        // RTW
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_rtw_kind(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_rtw_bytes(IntPtr r, IntPtr outBuf, ulong outCap, out ulong outLen);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_rtw_decode([In] byte[] bytes, ulong len);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rtorch_api_rtw_free(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_rtw_has_manifest(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_rtw_manifest_artifact_id(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_rtw_manifest_location(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_rtw_manifest_format_version(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_rtw_manifest_requires_count(IntPtr r);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_rtw_manifest_requires_dim(IntPtr r, ulong i);

        // Model
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_model_decode([In] byte[] payload, ulong len);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rtorch_api_model_free(IntPtr m);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_model_name(IntPtr m);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern uint rtorch_api_model_version(IntPtr m);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_model_num_params(IntPtr m);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_model_param_name(IntPtr m, ulong i);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_model_param_data(IntPtr m, ulong i, out ulong numel);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_model_param_shape_len(IntPtr m, ulong i);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_model_param_shape_dim(IntPtr m, ulong i, ulong j);

        // Memory
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_memory_decode([In] byte[] payload, ulong len);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rtorch_api_memory_free(IntPtr m);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_memory_num_frags(IntPtr m);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong rtorch_api_memory_frag_id(IntPtr m, ulong i);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rtorch_api_memory_frag_state(IntPtr m, ulong i, out ulong numel);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern float rtorch_api_memory_frag_strength(IntPtr m, ulong i);

        // Rule
        // The C ABI rule callback is `void (*)(const rtorch_api_blob* in,
        // size_t n_in, rtorch_api_blob* out, void* userdata)`. We take the blob
        // pointers as raw IntPtr (not managed Blob[]): the callback is invoked
        // from native with a C array / struct pointer, and blittable IntPtr avoids
        // the marshaler reinterpreting an array as a single struct pointer.
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int RuleFn(IntPtr inBlobs, ulong nIn, IntPtr outBlob, IntPtr userdata);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_register_rule([MarshalAs(UnmanagedType.LPStr)] string name, IntPtr fn, IntPtr userdata);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_rule_exists([MarshalAs(UnmanagedType.LPStr)] string name);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rtorch_api_run_rule([MarshalAs(UnmanagedType.LPStr)] string name, [In] Blob[] inBlobs, ulong nIn, [In, Out] Blob[] outBlob, IntPtr userdata);

        // ------------------------------------------------------------------
        // Last error.
        // ------------------------------------------------------------------
        public static string LastError()
        {
            IntPtr p = rtorch_api_last_error();
            return p == IntPtr.Zero ? "" : Marshal.PtrToStringAnsi(p) ?? "";
        }

        private static void Check(int rc, string what)
        {
            if (rc != OK)
                throw new RTorchApiException($"{what}: rc={rc} ({LastError()})", rc);
        }

        private static byte[] CopyBytes(IntPtr p, ulong len)
        {
            if (p == IntPtr.Zero || len == 0) return Array.Empty<byte>();
            var buf = new byte[len];
            Marshal.Copy(p, buf, 0, (int)len);
            return buf;
        }

        private static float[] CopyFloats(IntPtr p, ulong n)
        {
            if (p == IntPtr.Zero || n == 0) return Array.Empty<float>();
            var buf = new float[n];
            Marshal.Copy(p, buf, 0, (int)n);
            return buf;
        }

        // ------------------------------------------------------------------
        // Managed wrappers.
        // ------------------------------------------------------------------
        public sealed class Tensor : IDisposable
        {
            internal IntPtr _h;
            internal Tensor(IntPtr h) => _h = h;
            public Tensor(ulong[] dims, int dtype = DTYPE_F32)
            {
                _h = rtorch_api_tensor_new(dims, (ulong)dims.Length, dtype);
                if (_h == IntPtr.Zero) throw new RTorchApiException("tensor_new: null", E_PARAM);
            }
            public void Dispose() { if (_h != IntPtr.Zero) { rtorch_api_tensor_free(_h); _h = IntPtr.Zero; } }

            public ulong Numel => rtorch_api_tensor_numel(_h);
            public ulong Rank => rtorch_api_tensor_rank(_h);
            public ulong Dim(ulong i) => rtorch_api_tensor_dim(_h, i);
            public int Dtype => rtorch_api_tensor_dtype(_h);
            public float[] Data => CopyFloats(rtorch_api_tensor_data(_h), Numel);
            public ulong[] Shape { get { var s = new ulong[Rank]; for (ulong i = 0; i < Rank; i++) s[i] = Dim(i); return s; } }
        }

        public sealed class Session : IDisposable
        {
            internal IntPtr _h;
            internal Session(IntPtr h) => _h = h;
            public Session(string formulaSrc, string refDir = null)
            {
                _h = rtorch_api_session_compile(formulaSrc, refDir);
                if (_h == IntPtr.Zero) throw new RTorchApiException("session_compile: null", E_PARAM);
            }
            public void Dispose() { if (_h != IntPtr.Zero) { rtorch_api_session_free(_h); _h = IntPtr.Zero; } }

            public byte[] Run(IList<byte[]> inputs, int device = 0)
            {
                var blobs = new Blob[inputs.Count];
                for (int i = 0; i < inputs.Count; i++) blobs[i] = Blob.From(inputs[i]);
                // Probe output size.
                var probe = new Blob[] { new Blob(IntPtr.Zero, 0) };
                int rc = rtorch_api_session_execute(_h, blobs, (ulong)blobs.Length, probe, device);
                if (rc != OK && rc != E_PARAM) throw new RTorchApiException($"session_execute(probe): rc={rc}", rc);
                ulong want = probe[0].Len;
                var buf = new byte[Math.Max(want, 1)];
                var outBlob = new Blob[] { Blob.From(buf) };
                rc = rtorch_api_session_execute(_h, blobs, (ulong)blobs.Length, outBlob, device);
                if (rc != OK) throw new RTorchApiException($"session_execute: rc={rc}", rc);
                var res = new byte[outBlob[0].Len];
                Array.Copy(buf, res, (int)outBlob[0].Len);
                return res;
            }
        }

        public sealed class Rtw : IDisposable
        {
            internal IntPtr _h;
            internal Rtw(IntPtr h) => _h = h;
            public Rtw(byte[] payload)
            {
                _h = rtorch_api_rtw_decode(payload, (ulong)payload.Length);
                if (_h == IntPtr.Zero) throw new RTorchApiException("rtw_decode: null", E_PARSE);
            }
            public void Dispose() { if (_h != IntPtr.Zero) { rtorch_api_rtw_free(_h); _h = IntPtr.Zero; } }

            public int Kind => rtorch_api_rtw_kind(_h);
            public byte[] Bytes()
            {
                int rc = rtorch_api_rtw_bytes(_h, IntPtr.Zero, 0, out ulong len);
                if (rc != OK && rc != E_PARAM) throw new RTorchApiException($"rtw_bytes(probe): rc={rc}", rc);
                IntPtr buf = Marshal.AllocHGlobal((int)Math.Max(len, 1));
                try
                {
                    rc = rtorch_api_rtw_bytes(_h, buf, len, out ulong written);
                    if (rc != OK) throw new RTorchApiException($"rtw_bytes: rc={rc}", rc);
                    return CopyBytes(buf, written);
                }
                finally { Marshal.FreeHGlobal(buf); }
            }

            // RTW self-describing Manifest (RTorch 0.1.4).
            public bool HasManifest => rtorch_api_rtw_has_manifest(_h) != 0;
            public string ArtifactId => Marshal.PtrToStringAnsi(rtorch_api_rtw_manifest_artifact_id(_h)) ?? "";
            public string Location => Marshal.PtrToStringAnsi(rtorch_api_rtw_manifest_location(_h)) ?? "";
            public string FormatVersion => Marshal.PtrToStringAnsi(rtorch_api_rtw_manifest_format_version(_h)) ?? "";
            public ulong RequiresCount => rtorch_api_rtw_manifest_requires_count(_h);
            public string RequiresDim(ulong i) => Marshal.PtrToStringAnsi(rtorch_api_rtw_manifest_requires_dim(_h, i)) ?? "";
        }

        public sealed class Model : IDisposable
        {
            internal IntPtr _h;
            internal Model(IntPtr h) => _h = h;
            public Model(byte[] payload)
            {
                _h = rtorch_api_model_decode(payload, (ulong)payload.Length);
                if (_h == IntPtr.Zero) throw new RTorchApiException("model_decode: null", E_PARSE);
            }
            public void Dispose() { if (_h != IntPtr.Zero) { rtorch_api_model_free(_h); _h = IntPtr.Zero; } }

            public string Name => Marshal.PtrToStringAnsi(rtorch_api_model_name(_h)) ?? "";
            public uint Version => rtorch_api_model_version(_h);
            public ulong NumParams => rtorch_api_model_num_params(_h);
            public string ParamName(ulong i) => Marshal.PtrToStringAnsi(rtorch_api_model_param_name(_h, i)) ?? "";
            public float[] ParamData(ulong i) => CopyFloats(rtorch_api_model_param_data(_h, i, out ulong n), n);
            public ulong ParamShapeLen(ulong i) => rtorch_api_model_param_shape_len(_h, i);
            public ulong ParamShapeDim(ulong i, ulong j) => rtorch_api_model_param_shape_dim(_h, i, j);
        }

        public sealed class Memory : IDisposable
        {
            internal IntPtr _h;
            internal Memory(IntPtr h) => _h = h;
            public Memory(byte[] payload)
            {
                _h = rtorch_api_memory_decode(payload, (ulong)payload.Length);
                if (_h == IntPtr.Zero) throw new RTorchApiException("memory_decode: null", E_PARSE);
            }
            public void Dispose() { if (_h != IntPtr.Zero) { rtorch_api_memory_free(_h); _h = IntPtr.Zero; } }

            public ulong NumFrags => rtorch_api_memory_num_frags(_h);
            public ulong FragId(ulong i) => rtorch_api_memory_frag_id(_h, i);
            public float[] FragState(ulong i) => CopyFloats(rtorch_api_memory_frag_state(_h, i, out ulong n), n);
            public float FragStrength(ulong i) => rtorch_api_memory_frag_strength(_h, i);
        }

        // ------------------------------------------------------------------
        // Rule helpers.
        // ------------------------------------------------------------------
        private static readonly Dictionary<string, GCHandle> _ruleKeeps = new();

        public static int RegisterRule(string name, RuleFn fn, IntPtr userdata = default)
        {
            IntPtr fp = Marshal.GetFunctionPointerForDelegate(fn);
            _ruleKeeps[name] = GCHandle.Alloc(fn); // keep delegate pinned
            return rtorch_api_register_rule(name, fp, userdata);
        }

        public static bool RuleExists(string name) => rtorch_api_rule_exists(name) == 1;

        public static int RunRule(string name, IList<byte[]> inputs, out byte[] result, IntPtr userdata = default)
        {
            var blobs = new Blob[inputs.Count];
            for (int i = 0; i < inputs.Count; i++) blobs[i] = Blob.From(inputs[i]);
            // Allocate an unmanaged out buffer that the rule callback can write to.
            IntPtr outPtr = Marshal.AllocHGlobal(4096);
            var outBlob = new Blob[] { new Blob(outPtr, 4096) };
            try
            {
                int rc = rtorch_api_run_rule(name, blobs, (ulong)blobs.Length, outBlob, userdata);
                ulong written = outBlob[0].Len;
                result = rc == OK ? CopyBytes(outBlob[0].Data, written) : Array.Empty<byte>();
                return rc;
            }
            finally
            {
                // Free the unmanaged buffers from Blob.From(inputs) as well.
                for (int i = 0; i < blobs.Length; i++)
                    if (blobs[i].Data != IntPtr.Zero) Marshal.FreeHGlobal(blobs[i].Data);
                Marshal.FreeHGlobal(outPtr);
            }
        }
    }

    // The C ABI blob type (mirrors rtorch_api_blob).
    [StructLayout(LayoutKind.Sequential)]
    public struct Blob
    {
        public IntPtr Data;
        public ulong Len;

        public Blob(IntPtr data, ulong len) { Data = data; Len = len; }

        public static Blob From(byte[] bytes)
        {
            if (bytes == null || bytes.Length == 0) return new Blob(IntPtr.Zero, 0);
            IntPtr p = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, p, bytes.Length);
            return new Blob(p, (ulong)bytes.Length);
        }
    }

    public class RTorchApiException : Exception
    {
        public int Rc { get; }
        public RTorchApiException(string message, int rc) : base(message) { Rc = rc; }
    }
}
