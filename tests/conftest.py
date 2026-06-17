"""Shared pytest fixtures.

`torch_ga` and `versor` are imported lazily by fixture so tests that need
them are skipped cleanly if either witness library isn't installed.

Adds tests/ and _shared/ to sys.path. Helpers shared with the benches
(`_cayley`, `_einsum_refs`, `_rotor_apply_helpers`) live under _shared/
as a single canonical copy; tests-only helpers (`_gradcheck`) live here.
"""
import os
import sys

import pytest
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, os.pardir, "_shared")))
# VersorAI library (sibling of the repo): on path so correctness tests that
# use it as a witness can `import gacore.kernel`.
sys.path.insert(0, os.path.normpath(
    os.path.join(_HERE, os.pardir, os.pardir, "VersorAI", "library")))


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def torch_ga():
    return pytest.importorskip("torch_ga")


@pytest.fixture(scope="session")
def versor():
    # The Versor library installs as top-level packages `core`, `layers`, ...
    # `from core.algebra import CliffordAlgebra` is the entry point we need.
    core_algebra = pytest.importorskip("core.algebra")
    return core_algebra.CliffordAlgebra
