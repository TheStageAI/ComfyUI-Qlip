"""Test harness: import the node pack the way ComfyUI does, without a GPU,
without ComfyUI and without the licensed ``qlip`` package.

* ComfyUI (``comfy.*``, ``folder_paths``) is only imported lazily inside node
  methods, so importing the pack needs no ComfyUI.
* ``qlip`` is a licensed binary package that CI cannot install. The few qlip
  modules the nodes import at module level are replaced by permissive stubs
  (any attribute exists) BEFORE the pack is imported. Tests therefore cover
  the node layer — registration, schemas, host-side helpers — not qlip itself.
* torch is required (CPU build is enough).
"""

import importlib.util
import os
import sys
from unittest import mock

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "comfyui_qlip"

# qlip modules imported at module level by nodes/ and utils/
_QLIP_MODULES = (
    "qlip",
    "qlip.inference",
    "qlip.inference.step_cache",
    "qlip.inference.loom",
    "qlip.inference.loom.lora",
)


def _install_qlip_stubs():
    for name in _QLIP_MODULES:
        if name not in sys.modules:
            sys.modules[name] = mock.MagicMock(name=name)


def _load_pack():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    _install_qlip_stubs()
    spec = importlib.util.spec_from_file_location(
        PACKAGE,
        os.path.join(REPO, "__init__.py"),
        submodule_search_locations=[REPO],
    )
    pack = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = pack
    spec.loader.exec_module(pack)
    return pack


@pytest.fixture(scope="session")
def pack():
    """The imported node pack (``NODE_CLASS_MAPPINGS`` etc.)."""
    return _load_pack()


@pytest.fixture(scope="session")
def pack_module(pack):
    """Import a submodule of the pack: ``pack_module("nodes.compile")``."""

    def get(name):
        return importlib.import_module(f"{PACKAGE}.{name}")

    return get


@pytest.fixture(scope="session")
def repo_root():
    return REPO


def load_script(path, name):
    """Import a standalone script (tools/*.py, install.py) as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def script():
    def get(rel_path):
        name = "script_" + rel_path.replace("/", "_").replace(".py", "")
        return load_script(os.path.join(REPO, rel_path), name)

    return get
