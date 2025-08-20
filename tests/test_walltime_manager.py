"""Tests for walltime management and parsing."""

import pytest
import time
from sgorch.walltime_manager import (
    WalltimeManager,
    WalltimeInfo,
    parse_slurm_time_limit,
)


@pytest.fixture
def walltime_manager():
    """Create a walltime manager for testing."""
    return WalltimeManager("test-deployment", predrain_seconds=180)


class TestSlurmTimeParsing:
    """Tests for SLURM time limit parsing."""
    
    @pytest.mark.parametrize("time_str,expected_seconds", [
        # HH:MM:SS format
        ("01:30:45", 1 * 3600 + 30 * 60 + 45),
        ("00:05:00", 5 * 60),
        ("23:59:59", 23 * 3600 + 59 * 60 + 59),
        
        # HH:MM format (two parts, hours:minutes)
        ("05:30", 5 * 3600 + 30 * 60),
        ("12:15", 12 * 3600 + 15 * 60),
        ("23:45", 23 * 3600 + 45 * 60),
        
        # MM:SS format (two parts, minutes:seconds when first part > 23)
        ("90:30", 90 * 60 + 30),  # 90 minutes, 30 seconds
        ("120:00", 120 * 60),     # 120 minutes
        
        # DD-HH:MM:SS format (with days)
        ("1-00:00:00", 1 * 24 * 3600),
        ("2-12:30:45", 2 * 24 * 3600 + 12 * 3600 + 30 * 60 + 45),
        ("7-00:00:00", 7 * 24 * 3600),
        
        # Special cases
        ("INFINITE", 365 * 24 * 3600),
        ("UNLIMITED", 365 * 24 * 3600),
    ])
    def test_parse_slurm_time_limit_valid(self, time_str, expected_seconds):
        """Test parsing valid SLURM time limit strings."""
        result = parse_slurm_time_limit(time_str)
        assert result == expected_seconds
    
    def test_parse_slurm_time_limit_case_insensitive(self):
        """Test that infinite/unlimited parsing is case insensitive."""
        assert parse_slurm_time_limit("infinite") == 365 * 24 * 3600
        assert parse_slurm_time_limit("Unlimited") == 365 * 24 * 3600
        assert parse_slurm_time_limit("INFINITE") == 365 * 24 * 3600
    
    def test_parse_slurm_time_limit_invalid(self):
        """Test parsing invalid time limit strings."""
        with pytest.raises(ValueError):
            parse_slurm_time_limit("invalid")
        
        with pytest.raises(ValueError):
            parse_slurm_time_limit("25:70:90")  # Invalid minutes/seconds
        
        with pytest.raises(ValueError):
            parse_slurm_time_limit("")  # Empty string


class TestWalltimeInfo:
    """Tests for WalltimeInfo dataclass."""
    
    def test_walltime_info_creation(self, freeze_time):
        """Test WalltimeInfo creation and basic properties."""
        current_time = freeze_time.now()
        submitted_time = current_time - 1800  # 30 minutes ago
        time_limit = 3600  # 1 hour
        estimated_end = submitted_time + time_limit
        remaining = estimated_end - current_time
        
        info = WalltimeInfo(
            job_id="job123",
            time_limit_seconds=time_limit,
            submitted_at=submitted_time,
            estimated_end_time=estimated_end,
            time_remaining_seconds=remaining
        )
        
        assert info.job_id == "job123"
        assert info.time_limit_seconds == time_limit
        assert info.submitted_at == submitted_time
        assert info.estimated_end_time == estimated_end
        assert info.time_remaining_seconds == remaining
    
    def test_walltime_info_properties(self, freeze_time):
        """Test WalltimeInfo computed properties."""
        current_time = freeze_time.now()
        submitted_time = current_time - 1200  # 20 minutes ago
        time_limit = 3600  # 1 hour
        estimated_end = submitted_time + time_limit
        remaining = estimated_end - current_time  # 40 minutes remaining
        
        info = WalltimeInfo(
            job_id="job123",
            time_limit_seconds=time_limit,
            submitted_at=submitted_time,
            estimated_end_time=estimated_end,
            time_remaining_seconds=remaining
        )
        
        assert abs(info.minutes_remaining - 40) < 0.1  # 40 minutes remaining
        assert abs(info.percent_complete - 33.33) < 0.1  # 20/60 = 33.33%
    
    def test_is_approaching_walltime(self, freeze_time):
        """Test walltime approaching detection."""
        current_time = freeze_time.now()
        submitted_time = current_time - 3420  # 57 minutes ago
        time_limit = 3600  # 1 hour
        estimated_end = submitted_time + time_limit
        remaining = estimated_end - current_time  # 3 minutes remaining
        
        info = WalltimeInfo(
            job_id="job123",
            time_limit_seconds=time_limit,
            submitted_at=submitted_time,
            estimated_end_time=estimated_end,
            time_remaining_seconds=remaining
        )
        
        assert info.is_approaching_walltime(300)  # 5 minute threshold
        assert not info.is_approaching_walltime(120)  # 2 minute threshold


class TestWalltimeManager:
    """Tests for WalltimeManager class."""
    
    def test_walltime_manager_initialization(self, walltime_manager):
        """Test WalltimeManager initialization."""
        assert walltime_manager.deployment_name == "test-deployment"
        assert walltime_manager.predrain_seconds == 180
        assert len(walltime_manager.walltime_info) == 0
    
    def test_register_worker(self, walltime_manager, freeze_time):
        """Test worker registration with walltime info."""
        current_time = freeze_time.now()
        
        walltime_manager.register_worker("job123", "08:00:00", current_time)
        
        assert "job123" in walltime_manager.walltime_info
        info = walltime_manager.walltime_info["job123"]
        assert info.job_id == "job123"
        assert info.time_limit_seconds == 8 * 3600  # 8 hours
        assert info.submitted_at == current_time
    
    def test_register_worker_invalid_time(self, walltime_manager, freeze_time):
        """Test worker registration with invalid time limit."""
        # Should not crash, but should log a warning
        walltime_manager.register_worker("job123", "invalid-time", freeze_time.now())
        
        # Worker should not be registered
        assert "job123" not in walltime_manager.walltime_info
    
    def test_unregister_worker(self, walltime_manager, freeze_time):
        """Test worker unregistration."""
        walltime_manager.register_worker("job123", "08:00:00", freeze_time.now())
        assert "job123" in walltime_manager.walltime_info
        
        walltime_manager.unregister_worker("job123")
        assert "job123" not in walltime_manager.walltime_info
    
    def test_unregister_nonexistent_worker(self, walltime_manager):
        """Test unregistering a worker that doesn't exist."""
        # Should not raise an exception
        walltime_manager.unregister_worker("nonexistent")
    
    def test_update_remaining_times(self, walltime_manager, freeze_time):
        """Test updating remaining time calculations."""
        initial_time = freeze_time.now()
        walltime_manager.register_worker("job123", "01:00:00", initial_time)  # 1 hour limit
        
        # Initially, should have close to 1 hour remaining
        info = walltime_manager.walltime_info["job123"]
        assert abs(info.time_remaining_seconds - 3600) < 1
        
        # Advance time by 30 minutes
        freeze_time.advance(1800)
        walltime_manager.update_remaining_times()
        
        # Should have 30 minutes remaining
        info = walltime_manager.walltime_info["job123"]
        assert abs(info.time_remaining_seconds - 1800) < 1
        
        # Advance time beyond limit
        freeze_time.advance(3600)
        walltime_manager.update_remaining_times()
        
        # Should have 0 remaining (not negative)
        info = walltime_manager.walltime_info["job123"]
        assert info.time_remaining_seconds == 0
    
    def test_get_workers_approaching_walltime(self, walltime_manager, freeze_time):
        """Test finding workers approaching walltime."""
        current_time = freeze_time.now()
        
        # Register workers with different remaining times
        walltime_manager.register_worker("job1", "01:00:00", current_time - 3480)  # 2 min remaining (120s < 180s threshold)
        walltime_manager.register_worker("job2", "01:00:00", current_time - 1800)  # 30 min remaining  
        walltime_manager.register_worker("job3", "01:00:00", current_time - 3540)  # 1 min remaining (60s < 180s threshold)
        
        approaching = walltime_manager.get_workers_approaching_walltime()
        
        # Should find job1 and job3 (both < 180 seconds remaining)
        assert len(approaching) == 2
        job_ids = [w.job_id for w in approaching]
        assert "job1" in job_ids
        assert "job3" in job_ids
        assert "job2" not in job_ids
        
        # Should be sorted by urgency (least time remaining first)
        assert approaching[0].job_id == "job3"  # 1 minute remaining
        assert approaching[1].job_id == "job1"  # 5 minutes remaining
    
    def test_should_start_proactive_replacement(self, walltime_manager, freeze_time):
        """Test proactive replacement decision logic."""
        current_time = freeze_time.now()
        
        # Register worker with 2 minutes remaining (below 3-minute threshold)
        walltime_manager.register_worker("job1", "01:00:00", current_time - 3480)  # 3600-3480 = 120s = 2min
        
        assert walltime_manager.should_start_proactive_replacement("job1")
        
        # Worker with plenty of time should not trigger replacement
        walltime_manager.register_worker("job2", "08:00:00", current_time)
        assert not walltime_manager.should_start_proactive_replacement("job2")
        
        # Nonexistent worker should not trigger replacement
        assert not walltime_manager.should_start_proactive_replacement("nonexistent")
    
    @pytest.mark.parametrize("remaining_minutes,expected_urgency", [
        (2, "critical"),
        (10, "high"),
        (20, "medium"),
        (45, "low"),
    ])
    def test_get_replacement_urgency(self, walltime_manager, freeze_time, remaining_minutes, expected_urgency):
        """Test replacement urgency calculation."""
        current_time = freeze_time.now()
        time_limit_seconds = 3600  # 1 hour
        elapsed_seconds = time_limit_seconds - (remaining_minutes * 60)
        
        walltime_manager.register_worker("job1", "01:00:00", current_time - elapsed_seconds)
        
        urgency = walltime_manager.get_replacement_urgency("job1")
        assert urgency == expected_urgency
    
    def test_get_replacement_urgency_nonexistent(self, walltime_manager):
        """Test replacement urgency for nonexistent worker."""
        urgency = walltime_manager.get_replacement_urgency("nonexistent")
        assert urgency == "unknown"
    
    def test_estimate_replacement_window(self, walltime_manager, freeze_time):
        """Test replacement window estimation."""
        current_time = freeze_time.now()
        
        # Register worker with 30 minutes remaining
        walltime_manager.register_worker("job1", "01:00:00", current_time - 1800)
        
        window = walltime_manager.estimate_replacement_window("job1")
        
        assert window is not None
        assert abs(window['time_remaining_seconds'] - 1800) < 1  # 30 minutes
        assert window['estimated_replacement_time_seconds'] == 17 * 60  # 17 minutes
        assert window['time_buffer_seconds'] > 0  # Should have buffer
        assert window['can_complete_replacement'] is True
        assert window['urgency_level'] == "medium"
    
    def test_estimate_replacement_window_insufficient_time(self, walltime_manager, freeze_time):
        """Test replacement window when insufficient time remaining."""
        current_time = freeze_time.now()
        
        # Register worker with only 5 minutes remaining
        walltime_manager.register_worker("job1", "01:00:00", current_time - 3300)
        
        window = walltime_manager.estimate_replacement_window("job1")
        
        assert window is not None
        assert window['time_buffer_seconds'] < 0  # Not enough time
        assert window['can_complete_replacement'] is False
        assert window['urgency_level'] == "critical"
    
    def test_estimate_replacement_window_nonexistent(self, walltime_manager):
        """Test replacement window estimation for nonexistent worker."""
        window = walltime_manager.estimate_replacement_window("nonexistent")
        assert window is None
    
    def test_get_walltime_statistics_empty(self, walltime_manager):
        """Test walltime statistics with no workers."""
        stats = walltime_manager.get_walltime_statistics()
        assert stats == {}
    
    def test_get_walltime_statistics_with_workers(self, walltime_manager, freeze_time):
        """Test walltime statistics calculation."""
        current_time = freeze_time.now()
        
        # Register workers with different states
        walltime_manager.register_worker("job1", "01:00:00", current_time - 1800)  # 30 min remaining, 50% complete
        walltime_manager.register_worker("job2", "02:00:00", current_time - 3600)  # 60 min remaining, 50% complete
        walltime_manager.register_worker("job3", "01:00:00", current_time - 3480)  # 2 min remaining, approaching
        
        stats = walltime_manager.get_walltime_statistics()
        
        assert stats['workers_tracked'] == 3
        assert abs(stats['min_time_remaining_minutes'] - 2) < 0.1  # job3
        assert abs(stats['max_time_remaining_minutes'] - 60) < 0.1  # job2
        assert abs(stats['avg_time_remaining_minutes'] - 30.67) < 0.5  # (30+60+2)/3
        assert stats['workers_approaching_walltime'] == 1  # job3 only
    
    def test_walltime_manager_custom_predrain_time(self):
        """Test walltime manager with custom predrain time."""
        manager = WalltimeManager("test", predrain_seconds=600)  # 10 minutes
        assert manager.predrain_seconds == 600
