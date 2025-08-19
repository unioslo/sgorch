import json
import os
from pathlib import Path

from sgorch.state.file_store import FileStateStore
from sgorch.state.base import DeploymentSnapshot, SerializableWorker


def _snapshot():
    return DeploymentSnapshot(
        name="d",
        workers=[SerializableWorker(job_id="1", instance_uuid="u", node="n", remote_port=8000)],
        allocated_ports=[8000, 8001],
    )


def test_save_and_load_deployment_snapshot(tmp_path):
    f = tmp_path / "state.json"
    store = FileStateStore(file_path=str(f))
    snap = _snapshot()
    store.save_deployment(snap)
    loaded = store.load_deployment("d")
    assert loaded is not None
    assert loaded.name == "d"
    assert len(loaded.workers) == 1
    assert loaded.allocated_ports == [8000, 8001]


def test_corrupted_file_recovers_with_empty_state(tmp_path):
    f = tmp_path / "state.json"
    f.write_text("{ not json }")
    store = FileStateStore(file_path=str(f))
    assert store.load_deployment("d") is None
    # saving should overwrite
    store.save_deployment(_snapshot())
    assert store.load_deployment("d") is not None


def test_delete_deployment_removes_entry(tmp_path):
    f = tmp_path / "state.json"
    store = FileStateStore(file_path=str(f))
    store.save_deployment(_snapshot())
    assert store.load_deployment("d") is not None
    store.delete_deployment("d")
    assert store.load_deployment("d") is None


def test_env_var_overrides_path(tmp_path, monkeypatch):
    f = tmp_path / "env.json"
    monkeypatch.setenv("SGORCH_STATE_FILE", str(f))
    store = FileStateStore()
    store.save_deployment(_snapshot())
    assert f.exists()

