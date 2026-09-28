"""Minimal CUDA runtime-JIT runner for Windows without MSVC.

This machine has the CUDA 11.6 toolkit (nvcc + NVRTC) but no host C++ compiler
(cl.exe), so torch.utils.cpp_extension cannot build anything locally. NVRTC does
not need a host compiler, so we compile the pure-CUDA source at run time and
launch it through the CUDA driver API via ctypes.

The source compiled here is byte-identical to the CUDA source embedded in
submission.py (we import the same string), so local measurements exercise the
same kernels that the official runner will build with nvcc.

Only builtins are used (__shfl_up_sync, float4, __syncthreads, ...), so no
include paths or headers are required.
"""
import ctypes
import os
import sys

import torch

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin"


def _load(name, path_hint=None):
    if path_hint:
        os.add_dll_directory(path_hint)
        try:
            return ctypes.WinDLL(os.path.join(path_hint, name))
        except OSError:
            pass
    return ctypes.WinDLL(name)


_nvrtc = None
_cuda = None


def _nvrtc_lib():
    global _nvrtc
    if _nvrtc is None:
        _nvrtc = _load("nvrtc64_112_0.dll", CUDA_BIN)
        _nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
        _nvrtc.nvrtcCreateProgram.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                              ctypes.c_char_p, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_void_p]
        _nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
        _nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_char_p)]
        _nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
        _nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        _nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
        _nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _nvrtc.nvrtcGetCUBINSize.restype = ctypes.c_int
        _nvrtc.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        _nvrtc.nvrtcGetCUBIN.restype = ctypes.c_int
        _nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
        _nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        _nvrtc.nvrtcGetPTX.restype = ctypes.c_int
        _nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    return _nvrtc


def _cuda_lib():
    global _cuda
    if _cuda is None:
        _cuda = _load("nvcuda.dll")
        _cuda.cuInit.restype = ctypes.c_int
        _cuda.cuInit.argtypes = [ctypes.c_uint]
        _cuda.cuDeviceGet.restype = ctypes.c_int
        _cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        _cuda.cuModuleLoadData.restype = ctypes.c_int
        _cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _cuda.cuModuleGetFunction.restype = ctypes.c_int
        _cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_void_p, ctypes.c_char_p]
        _cuda.cuLaunchKernel.restype = ctypes.c_int
        _cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + \
            [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return _cuda


def _check(code, what):
    if code != 0:
        raise RuntimeError("%s failed with CUDA error %d" % (what, code))


_initialised = False


def init():
    global _initialised
    if _initialised:
        return
    # make sure torch has created/attached the primary context
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    _check(_cuda_lib().cuInit(0), "cuInit")
    dev = ctypes.c_int(0)
    _check(_cuda_lib().cuDeviceGet(ctypes.byref(dev), torch.cuda.current_device()), "cuDeviceGet")
    _initialised = True


def compile_cuda(source, name="prefixsum.cu", arch=None, want_cubin=True, verbose=False):
    """NVRTC-compile CUDA source -> cubin bytes (falls back to PTX)."""
    lib = _nvrtc_lib()
    if arch is None:
        major, minor = torch.cuda.get_device_capability(0)
        arch = "sm_%d%d" % (major, minor)

    prog = ctypes.c_void_p()
    src = source.encode()
    _check(lib.nvrtcCreateProgram(ctypes.byref(prog), src, name.encode(), 0, None, None),
           "nvrtcCreateProgram")

    # note: no --use_fast_math; the official build uses default nvcc flags and we
    # must keep the numerics identical to what the platform will run.
    opts = ["--gpu-architecture=" + arch, "-std=c++17", "--device-debug" if False else "-lineinfo"]
    carr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = lib.nvrtcCompileProgram(prog, len(opts), carr)

    sz = ctypes.c_size_t()
    _check(lib.nvrtcGetProgramLogSize(prog, ctypes.byref(sz)), "logSize")
    log = ctypes.create_string_buffer(sz.value + 1)
    lib.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise RuntimeError("NVRTC compilation failed:\n" + log.value.decode(errors="replace"))
    if verbose and log.value:
        print(log.value.decode(errors="replace"))

    kind = "cubin"
    if want_cubin:
        if lib.nvrtcGetCUBINSize(prog, ctypes.byref(sz)) != 0 or sz.value == 0:
            want_cubin = False
    if want_cubin:
        buf = ctypes.create_string_buffer(sz.value)
        _check(lib.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
    else:
        kind = "ptx"
        _check(lib.nvrtcGetPTXSize(prog, ctypes.byref(sz)), "nvrtcGetPTXSize")
        buf = ctypes.create_string_buffer(sz.value)
        _check(lib.nvrtcGetPTX(prog, buf), "nvrtcGetPTX")
    return buf.raw, kind, arch


class Module:
    def __init__(self, source, verbose=False):
        init()
        blob, kind, arch = compile_cuda(source, verbose=verbose)
        self.kind, self.arch = kind, arch
        self._buf = ctypes.create_string_buffer(blob, len(blob))  # keep alive
        self.handle = ctypes.c_void_p()
        _check(_cuda_lib().cuModuleLoadData(ctypes.byref(self.handle), self._buf), "cuModuleLoadData")
        self._funcs = {}

    def func(self, name):
        if name not in self._funcs:
            h = ctypes.c_void_p()
            _check(_cuda_lib().cuModuleGetFunction(ctypes.byref(h), self.handle, name.encode()),
                   "cuModuleGetFunction(%s)" % name)
            self._funcs[name] = h
        return self._funcs[name]

    def launch(self, name, grid, block, args, shared=0, stream=None):
        """args: list of python values (int -> c_int, float -> c_float, tensor -> pointer)."""
        cvals = []
        for a in args:
            if isinstance(a, torch.Tensor):
                cvals.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, int):
                cvals.append(ctypes.c_int(a))
            elif isinstance(a, float):
                cvals.append(ctypes.c_float(a))
            elif isinstance(a, ctypes.c_void_p):
                cvals.append(a)
            else:
                raise TypeError("unsupported arg %r" % (a,))
        params = (ctypes.c_void_p * len(cvals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in cvals])
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        gx, gy, gz = (list(grid) + [1, 1, 1])[:3] if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        bx, by, bz = (list(block) + [1, 1, 1])[:3] if isinstance(block, (tuple, list)) else (block, 1, 1)
        _check(_cuda_lib().cuLaunchKernel(self.func(name), gx, gy, gz, bx, by, bz,
                                          shared, ctypes.c_void_p(stream),
                                          ctypes.cast(params, ctypes.c_void_p), None),
               "cuLaunchKernel(%s)" % name)


if __name__ == "__main__":
    smoke = r"""
extern "C" __global__ void addone(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
"""
    m = Module(smoke, verbose=True)
    print("compiled:", m.kind, m.arch)
    t = torch.zeros(8, device="cuda")
    m.launch("addone", grid=1, block=32, args=[t, 8])
    torch.cuda.synchronize()
    print("result:", t.cpu().tolist())
    assert t.sum().item() == 8
    print("NVRTC smoke test OK")
