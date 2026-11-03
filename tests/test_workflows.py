"""Shipped workflows: valid JSON, and every Qlip node they use exists."""

import glob
import json
import os

import pytest


def _workflows(repo_root):
    return sorted(
        glob.glob(os.path.join(repo_root, "workflows", "**", "*.json"), recursive=True)
    )


def _node_types(wf):
    if "nodes" in wf:  # UI format
        return [n.get("type") for n in wf["nodes"]]
    return [n.get("class_type") for n in wf.values() if isinstance(n, dict)]  # API


def test_workflows_exist(repo_root):
    assert _workflows(repo_root)


def test_workflow_qlip_nodes_are_registered(pack, repo_root):
    known = set(pack.NODE_CLASS_MAPPINGS)
    for path in _workflows(repo_root):
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        used = {
            t for t in _node_types(wf) if isinstance(t, str) and t.startswith("Qlip")
        }
        missing = used - known
        assert not missing, f"{os.path.relpath(path, repo_root)}: {sorted(missing)}"


@pytest.mark.parametrize("name", ["z-image-turbo-api.json"])
def test_api_workflow_links_resolve(repo_root, name):
    path = os.path.join(repo_root, "workflows", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not in this checkout")
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    for nid, node in wf.items():
        for inp, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2:
                assert str(value[0]) in wf, f"{nid}.{inp} -> missing node {value[0]}"
