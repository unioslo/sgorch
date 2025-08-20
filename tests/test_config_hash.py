"""Tests for configuration hashing and change detection."""

import pytest
import time
from pathlib import Path
from sgorch.config import load_config, DeploymentConfig
from sgorch.config_hash import (
    ConfigSnapshot,
    WorkerGeneration,
    _hash_dict,
)


@pytest.fixture
def sample_deployment_config(tmp_path):
    """Create a sample deployment config for testing."""
    config_yaml = tmp_path / "test_config.yaml"
    config_yaml.write_text("""
        deployments:
          - name: test-deployment
            replicas: 2
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: localhost
              advertise_host: 127.0.0.1
              local_port_range: [30000, 30010]
            router:
              base_url: http://router
            slurm:
              prefer: cli
              account: test-account
              partition: test-partition
              gres: "gpu:2"
              cpus_per_task: 16
              mem: "64GB"
              time_limit: "08:00:00"
              env:
                CUDA_VISIBLE_DEVICES: "0,1"
                TORCH_CUDA_ARCH_LIST: "8.0"
              log_dir: /tmp/logs
            sglang:
              model_path: /path/to/model
              args: ["--host", "0.0.0.0", "--port", "8000"]
              venv_path: /path/to/venv
            health:
              path: /health
              interval_s: 10
              timeout_s: 5
              consecutive_ok_for_ready: 3
              failures_to_unhealthy: 2
    """)
    
    config = load_config(config_yaml)
    return config.deployments[0].expand_variables()


def test_hash_dict_consistency():
    """Test that hash_dict produces consistent results."""
    data = {
        "key1": "value1",
        "key2": {"nested": "value"},
        "key3": ["list", "item"]
    }
    
    hash1 = _hash_dict(data)
    hash2 = _hash_dict(data)
    
    assert hash1 == hash2
    assert len(hash1) == 16  # Should be 16 character hex string
    

def test_hash_dict_order_independence():
    """Test that hash_dict is independent of key order."""
    data1 = {"b": 2, "a": 1, "c": 3}
    data2 = {"a": 1, "c": 3, "b": 2}
    
    hash1 = _hash_dict(data1)
    hash2 = _hash_dict(data2)
    
    assert hash1 == hash2


def test_hash_dict_sensitivity():
    """Test that hash_dict detects small changes."""
    data1 = {"key": "value1"}
    data2 = {"key": "value2"}
    
    hash1 = _hash_dict(data1)
    hash2 = _hash_dict(data2)
    
    assert hash1 != hash2


def test_config_snapshot_creation(sample_deployment_config):
    """Test ConfigSnapshot creation from deployment config."""
    snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    
    # Check SLURM fields
    assert snapshot.gres == "gpu:2"
    assert snapshot.cpus_per_task == 16
    assert snapshot.mem == "64GB"
    assert snapshot.time_limit == "08:00:00"
    assert snapshot.account == "test-account"
    assert snapshot.partition == "test-partition"
    assert snapshot.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    
    # Check SGLang fields
    assert snapshot.model_path == "/path/to/model"
    assert "--host" in snapshot.sglang_args
    assert snapshot.venv_path == "/path/to/venv"
    
    # Check runtime fields
    assert snapshot.connectivity_mode == "direct"
    assert snapshot.health_config_hash is not None


def test_config_snapshot_hash_consistency(sample_deployment_config):
    """Test that config snapshot hashes are consistent."""
    snapshot1 = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    snapshot2 = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    
    hash1 = snapshot1.compute_hash()
    hash2 = snapshot2.compute_hash()
    
    assert hash1 == hash2
    assert len(hash1) == 16


def test_config_snapshot_change_detection(sample_deployment_config, tmp_path):
    """Test that config snapshot detects changes."""
    # Create original snapshot
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    original_hash = original_snapshot.compute_hash()
    
    # Create modified config with different GRES
    modified_config_yaml = tmp_path / "modified_config.yaml"
    modified_config_yaml.write_text("""
        deployments:
          - name: test-deployment
            replicas: 2
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: localhost
              advertise_host: 127.0.0.1
              local_port_range: [30000, 30010]
            router:
              base_url: http://router
            slurm:
              prefer: cli
              account: test-account
              partition: test-partition
              gres: "gpu:4"  # Changed from gpu:2
              cpus_per_task: 16
              mem: "64GB"
              time_limit: "08:00:00"
              env:
                CUDA_VISIBLE_DEVICES: "0,1"
                TORCH_CUDA_ARCH_LIST: "8.0"
              log_dir: /tmp/logs
            sglang:
              model_path: /path/to/model
              args: ["--host", "0.0.0.0", "--port", "8000"]
              venv_path: /path/to/venv
            health:
              path: /health
              interval_s: 10
              timeout_s: 5
              consecutive_ok_for_ready: 3
              failures_to_unhealthy: 2
    """)
    
    modified_config = load_config(modified_config_yaml)
    modified_deployment = modified_config.deployments[0].expand_variables()
    modified_snapshot = ConfigSnapshot.from_deployment_config(modified_deployment)
    modified_hash = modified_snapshot.compute_hash()
    
    assert original_hash != modified_hash
    assert original_snapshot.requires_job_replacement(modified_snapshot)


def test_requires_job_replacement_scenarios(sample_deployment_config):
    """Test various scenarios for job replacement requirements."""
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    
    # Test SLURM resource changes (should require replacement)
    test_cases = [
        ("gres", "gpu:4"),
        ("cpus_per_task", 32),
        ("mem", "128GB"),
        ("model_path", "/new/model/path"),
    ]
    
    for field, new_value in test_cases:
        modified_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
        setattr(modified_snapshot, field, new_value)
        
        assert original_snapshot.requires_job_replacement(modified_snapshot), \
            f"Should require replacement for {field} change"


def test_requires_health_update_only(sample_deployment_config):
    """Test changes that only require health monitoring update."""
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    
    # Create modified snapshot with different health config
    modified_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    modified_snapshot.health_config_hash = "different_hash"
    
    assert not original_snapshot.requires_job_replacement(modified_snapshot)
    assert original_snapshot.requires_health_update(modified_snapshot)


def test_worker_generation_creation(sample_deployment_config, freeze_time):
    """Test WorkerGeneration creation and properties."""
    snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    config_hash = snapshot.compute_hash()
    
    generation = WorkerGeneration(
        config_hash=config_hash,
        config_snapshot=snapshot,
        created_at=freeze_time.now(),
        generation_id="test-gen-1"
    )
    
    assert generation.config_hash == config_hash
    assert generation.config_snapshot == snapshot
    assert generation.created_at == freeze_time.now()
    assert generation.generation_id == "test-gen-1"


def test_worker_generation_needs_replacement(sample_deployment_config):
    """Test WorkerGeneration replacement detection."""
    # Create original generation
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    generation = WorkerGeneration(
        config_hash=original_snapshot.compute_hash(),
        config_snapshot=original_snapshot,
        created_at=time.time(),
        generation_id="test-gen-1"
    )
    
    # Test with same config (should not need replacement)
    same_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    assert not generation.needs_replacement(same_snapshot)
    
    # Test with different config (should need replacement)
    different_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    different_snapshot.gres = "gpu:4"  # Change GRES
    assert generation.needs_replacement(different_snapshot)


def test_worker_generation_health_update_detection(sample_deployment_config):
    """Test WorkerGeneration health update detection."""
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    generation = WorkerGeneration(
        config_hash=original_snapshot.compute_hash(),
        config_snapshot=original_snapshot,
        created_at=time.time(),
        generation_id="test-gen-1"
    )
    
    # Test with same health config
    same_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    assert not generation.needs_health_update(same_snapshot)
    
    # Test with different health config
    different_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    different_snapshot.health_config_hash = "different_hash"
    assert generation.needs_health_update(different_snapshot)


def test_multiple_field_changes(sample_deployment_config):
    """Test detection of multiple simultaneous changes."""
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    
    # Create snapshot with multiple changes
    modified_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    modified_snapshot.gres = "gpu:4"
    modified_snapshot.cpus_per_task = 32
    modified_snapshot.mem = "128GB"
    modified_snapshot.model_path = "/new/model"
    
    assert original_snapshot.requires_job_replacement(modified_snapshot)
    
    # Create generation and test detailed reason
    generation = WorkerGeneration(
        config_hash=original_snapshot.compute_hash(),
        config_snapshot=original_snapshot,
        created_at=time.time(),
        generation_id="test-gen-1"
    )
    
    assert generation.needs_replacement(modified_snapshot)


@pytest.mark.parametrize("field,old_value,new_value", [
    ("gres", "gpu:2", "gpu:4"),
    ("cpus_per_task", 16, 32),
    ("mem", "64GB", "128GB"),
    ("time_limit", "08:00:00", "12:00:00"),
    ("account", "test-account", "new-account"),
    ("partition", "test-partition", "new-partition"),
    ("model_path", "/path/to/model", "/new/model"),
])
def test_individual_field_changes(sample_deployment_config, field, old_value, new_value):
    """Test that individual field changes are detected correctly."""
    original_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    
    # Verify the field has the expected old value
    assert getattr(original_snapshot, field) == old_value
    
    # Create modified snapshot
    modified_snapshot = ConfigSnapshot.from_deployment_config(sample_deployment_config)
    setattr(modified_snapshot, field, new_value)
    
    # Should require job replacement for all these fields
    assert original_snapshot.requires_job_replacement(modified_snapshot)
    
    # Hashes should be different
    assert original_snapshot.compute_hash() != modified_snapshot.compute_hash()
