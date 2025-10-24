#!/usr/bin/env python3
"""
Comprehensive GPU troubleshooting script.

Checks:
 1) PCIe GPU detection (lspci)
 2) NVIDIA driver/NVML status (nvidia-smi, modinfo, /proc/driver/nvidia)
 3) Container runtime configuration hints (cgroup, env, dev nodes)
 4) CUDA environment variables (CUDA_HOME, PATH, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES)
 5) PyTorch GPU binding (torch.cuda.*, versions)

Outputs clear PASS/FAIL for each section to help diagnose why a GPU (e.g., RTX 4090) is not accessible.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out
    except subprocess.CalledProcessError as e:
        return False, e.output
    except FileNotFoundError as e:
        return False, str(e)


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def section_pci():
    header("1) PCIe GPU Hardware Detection (lspci)")
    ok, out = run(["lspci"])
    if not ok:
        print("FAIL: lspci not available or failed:\n" + out.strip())
        return False
    gpu_lines = [l for l in out.splitlines() if ("NVIDIA" in l or "VGA compatible controller" in l or "3D controller" in l)]
    if gpu_lines:
        print("PASS: lspci output shows GPU-related devices:")
        for l in gpu_lines:
            print("  ", l)
        return True
    else:
        print("FAIL: No NVIDIA/3D controller entries found in lspci output")
        return False


def section_driver():
    header("2) NVIDIA Driver / NVML Status")
    ok, out = run(["nvidia-smi"])
    if ok:
        print("PASS: nvidia-smi output:")
        print(out)
        return True
    else:
        print("WARN: nvidia-smi failed:\n" + out.strip())
    # Check /proc and modinfo for hints
    proc_path = Path("/proc/driver/nvidia/version")
    if proc_path.exists():
        print("INFO: /proc/driver/nvidia/version present:")
        try:
            print(proc_path.read_text())
        except Exception as e:
            print("  (unable to read)", e)
    else:
        print("INFO: /proc/driver/nvidia/version not present")
    if shutil.which("modinfo"):
        ok_mi, out_mi = run(["modinfo", "nvidia"])
        if ok_mi:
            print("INFO: modinfo nvidia (kernel module present)")
            print(out_mi)
        else:
            print("INFO: modinfo nvidia failed (module not present?):")
            print(out_mi.strip())
    else:
        print("INFO: modinfo not available")
    # Device nodes
    dev_nodes = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
    found_nodes = [p for p in dev_nodes if Path(p).exists()]
    if found_nodes:
        print("PASS: NVIDIA device nodes present:")
        for p in found_nodes:
            print("  ", p)
    else:
        print("FAIL: No NVIDIA device nodes found under /dev")
    return False


def section_container():
    header("3) Container Runtime Configuration")
    # cgroup info
    ok, out = run(["cat", "/proc/1/cgroup"])
    if ok:
        print("INFO: /proc/1/cgroup:")
        print(out)
    else:
        print("INFO: Could not read /proc/1/cgroup")
    # environment variables that suggest GPU runtime
    env_checks = [
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
    ]
    for k in env_checks:
        v = os.environ.get(k)
        print(f"ENV {k} = {v}")
    # container runtime binaries
    for bin in ("docker", "nvidia-container-runtime", "nvidia-container-cli"):
        print(f"which {bin} ->", shutil.which(bin))
    print("Hint: If running in Docker, ensure you start the container with --gpus all and the NVIDIA Container Toolkit installed.")


def section_cuda_env():
    header("4) CUDA Environment Variables")
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    print("PATH:", os.environ.get("PATH"))
    # nvcc version if available
    if shutil.which("nvcc"):
        ok, out = run(["nvcc", "--version"])
        print("nvcc --version:")
        print(out.strip())
    else:
        print("nvcc not found in PATH")


def section_pytorch():
    header("5) PyTorch GPU Binding")
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


def main():
    print("GPU Troubleshooting Report")
    pci_ok = section_pci()
    section_driver()
    section_container()
    section_cuda_env()
    section_pytorch()

    print("\nSummary/Hints:")
    print("- If lspci shows NVIDIA GPU but nvidia-smi fails: host driver may be missing or not mounted into container.")
    print("- If /dev/nvidia* nodes are missing inside container: start with --gpus all and NVIDIA Container Toolkit.")
    print("- If torch sees no GPU: ensure PyTorch CUDA build matches driver/runtime and devices are visible (CUDA_VISIBLE_DEVICES).")
    print("- For RTX 4090 (Ada), driver >= 520.x typically required; verify host driver version with nvidia-smi.")


if __name__ == "__main__":
    main()

