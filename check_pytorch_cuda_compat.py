#!/usr/bin/env python3
"""
Checks to troubleshoot PyTorch-CUDA connection when GPU/driver exists but torch can't use it.

It reports:
 1) NVIDIA driver version vs PyTorch CUDA version (compat guidance)
 2) LD_LIBRARY_PATH presence of CUDA/NVIDIA libs and attempts to locate critical .so files
 3) PyTorch installation method (pip vs conda) and build info
 4) CUDA runtime libraries discoverable on the system

It prints actionable hints to align driver, libraries, and the torch build.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out
    except Exception as e:
        return False, str(e)


def parse_driver_version(text: str):
    m = re.search(r"NVRM version:\s*NVIDIA .*? (\d+\.\d+)", text)
    if m:
        return m.group(1)
    # fallback from nvidia-smi if available
    m = re.search(r"Driver Version:\s*(\d+\.\d+)", text)
    if m:
        return m.group(1)
    return None


def section_driver_vs_torch():
    print("== Driver vs PyTorch CUDA build ==")
    ok, out = run(["cat", "/proc/driver/nvidia/version"])
    drv = parse_driver_version(out) if ok else None
    if not drv:
        ok2, smi = run(["nvidia-smi"])
        if ok2:
            drv = parse_driver_version(smi)
    try:
        import torch
        torch_cuda = getattr(torch.version, "cuda", None)
        torch_ver = torch.__version__
    except Exception as e:
        print("PyTorch import failed:", e)
        return
    print(f"Driver version: {drv or 'unknown'}")
    print(f"PyTorch version: {torch_ver}")
    print(f"PyTorch CUDA: {torch_cuda}")
    # Rough guidance: CUDA 12.1 typically requires >= 525+; Ada (4090) needs >= 520+
    if torch_cuda:
        hints = []
        try:
            major_minor = tuple(map(int, torch_cuda.split(".")))
            if major_minor >= (12, 0):
                hints.append("CUDA 12.x generally requires NVIDIA driver >= 525.x")
            elif major_minor >= (11, 8):
                hints.append("CUDA 11.8 requires driver >= 520.x")
        except Exception:
            pass
        if drv:
            try:
                d_major = int(drv.split(".")[0])
                if (major_minor >= (12, 0) and d_major < 525) or (major_minor >= (11, 8) and d_major < 520):
                    hints.append("Driver likely too old for this CUDA runtime")
            except Exception:
                pass
        if hints:
            print("Hints:")
            for h in hints:
                print(" -", h)


def section_ld_library_path():
    print("\n== LD_LIBRARY_PATH and CUDA libs ==")
    ld = os.environ.get("LD_LIBRARY_PATH")
    print("LD_LIBRARY_PATH:", ld)
    search_paths = []
    if ld:
        search_paths.extend([p for p in ld.split(":") if p])
    # common defaults
    search_paths.extend([
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/wsl/lib",
    ])
    search_paths = [p for p in dict.fromkeys(search_paths)]
    needed = ["libcuda.so", "libcudart.so", "libcublas.so", "libnvrtc.so", "libnvToolsExt.so"]
    for lib in needed:
        found = []
        for p in search_paths:
            try:
                for f in Path(p).glob(lib + "*"):
                    found.append(str(f))
            except Exception:
                pass
        print(f"{lib}: {'FOUND' if found else 'MISSING'}")
        for f in found[:5]:
            print("  ", f)


def section_torch_build():
    print("\n== PyTorch install/build info ==")
    try:
        import torch
        print("torch.__version__:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        # pip vs conda heuristic
        import sys
        print("python:", sys.version)
        # show torch file location to guess installer
        import inspect
        loc = inspect.getfile(torch)
        print("torch module path:", loc)
        # show pip/conda if present
        pip = shutil.which("pip")
        conda = shutil.which("conda")
        print("pip:", pip)
        print("conda:", conda)
        if conda:
            ok, out = run(["conda", "list", "pytorch"]) 
            if ok:
                print("conda list pytorch:\n" + out)
        if pip:
            ok, out = run(["pip", "show", "torch"]) 
            if ok:
                print("pip show torch:\n" + out)
    except Exception as e:
        print("PyTorch import failed:", e)


def section_cuda_runtime():
    print("\n== CUDA runtime binaries and libs ==")
    bins = ["nvcc", "nvidia-smi", "nvidia-smi"]
    for b in bins:
        print(f"which {b} ->", shutil.which(b))
    ok, out = run(["nvcc", "--version"]) 
    print("nvcc --version:\n" + (out.strip() if ok else out))


def final_hints():
    print("\n== Actionable Fix Hints ==")
    print("- Ensure the container/session exposes the GPU nodes (e.g., Docker --gpus all).")
    print("- Driver must be new enough for your CUDA: e.g., CUDA 12.x -> driver >= 525.x; Ada (RTX 4090) -> >= 520.x.")
    print("- Prefer matching torch build to driver/runtime (e.g., pip torch+cu121 on host with >=525 driver).")
    print("- If LD_LIBRARY_PATH misses NVIDIA libs, add /usr/local/cuda/lib64 and system lib paths.")
    print("- If using conda, avoid mixing pip and conda CUDA toolkits; use a consistent environment.")
    print("- Verify /dev/nvidia0 exists and is accessible (check permissions/cgroups).")


def main():
    section_driver_vs_torch()
    section_ld_library_path()
    section_torch_build()
    section_cuda_runtime()
    final_hints()


if __name__ == "__main__":
    main()

