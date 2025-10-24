#!/usr/bin/env python3
"""
GPU Diagnostic: prints NVIDIA driver/SMI status, CUDA toolkit/runtime versions,
and PyTorch CUDA detection details.
"""

import subprocess
import shutil
import sys

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out
    except subprocess.CalledProcessError as e:
        return False, e.output
    except FileNotFoundError as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("NVIDIA SMI")
    print("=" * 80)
    ok, out = run(["nvidia-smi"])
    if ok:
        print(out)
    else:
        print("nvidia-smi not available or failed:")
        print(out.strip())

    print("\n" + "=" * 80)
    print("CUDA Toolkit (nvcc)")
    print("=" * 80)
    if shutil.which("nvcc"):
        ok, out = run(["nvcc", "--version"])
        print(out.strip())
    else:
        print("nvcc not found in PATH")

    print("\n" + "=" * 80)
    print("PyTorch CUDA Detection")
    print("=" * 80)
    try:
        import torch
        print("torch.__version__:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        try:
            avail = torch.cuda.is_available()
        except Exception as e:
            print("torch.cuda.is_available() raised:", repr(e))
            avail = False
        print("torch.cuda.is_available():", avail)
        if avail:
            try:
                count = torch.cuda.device_count()
                print("cuda device count:", count)
                for i in range(count):
                    try:
                        print(f"device {i}:", torch.cuda.get_device_name(i))
                        print("capability:", torch.cuda.get_device_capability(i))
                    except Exception as e:
                        print(f"error querying device {i}:", repr(e))
            except Exception as e:
                print("error getting device count:", repr(e))
        else:
            print("No CUDA device visible to PyTorch")
    except Exception as e:
        print("PyTorch import failed:", repr(e))


if __name__ == "__main__":
    main()

