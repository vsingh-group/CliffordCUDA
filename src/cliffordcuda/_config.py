"""Package-internal paths for kernel sources and the torch JIT cache.

`_GA_KERNELS_DIR` and `_ROTOR_KERNELS_DIR` resolve to the `.cu` directories
inside the installed package (they ship via `package-data`). The torch
extension build cache is placed under the user's XDG cache directory so we
never write into a read-only install (e.g. site-packages).
"""
import os
import time
from pathlib import Path

_PACKAGE_DIR       = Path(__file__).resolve().parent
_GA_KERNELS_DIR    = str(_PACKAGE_DIR / "kernels" / "ga")
_ROTOR_KERNELS_DIR = str(_PACKAGE_DIR / "kernels" / "rotor")
_DATA_DIR          = _PACKAGE_DIR / "_data"

# User-writable cache (XDG: $XDG_CACHE_HOME or ~/.cache). Used for both the
# torch JIT extension build directory and for runtime-computed perm files
# (see _utils.perm).
_USER_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "cliffordcuda"

os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR", str(_USER_CACHE_DIR / "torch_extensions")
)


def load_extension(name, sources, **kwargs):
    """Wrap torch.utils.cpp_extension.load with a status print."""
    from torch.utils.cpp_extension import load
    print(f"  {name} ...", flush=True)
    t0 = time.perf_counter()
    mod = load(name=name, sources=sources, **kwargs)
    elapsed = time.perf_counter() - t0
    verb = "compiled" if elapsed > 2.0 else "loaded"
    print(f"  {name} {verb} in {elapsed:.1f}s", flush=True)
    return mod
