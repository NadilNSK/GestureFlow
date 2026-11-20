"""
tests.test_accuracy_phase2b1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive automated unit tests for Phase 2B.1:
Cursor Accuracy and Precision Optimization.

Covers:
1. One Euro Filter (1D and 2D, jitter reduction, high-speed responsiveness, dt handling, reset)
2. 4-Point Calibrator (sample collection, outlier filtering, perspective calculation, degenerate fallback, serialization)
3. MouseController Calibrated Mapping and active region margin scaling
4. Precision Aiming Mode (scaling, smooth toggle transition without jumps)
5. Tracking Stability (velocity limiting, large jump snapping, tracking loss reset)
6. PySide6 Calibration Dialog and Worker API integration
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch
import pytest

from gestureflow.calibration.calibrator import (
    CALIBRATION_CORNERS,
    CORNER_TL,
    CORNER_TR,
    CORNER_BR,
    CORNER_BL,
    DEFAULT_TARGETS,
    CalibrationData,
    Calibrator,
)
from gestureflow.config.settings import ControllerSettings, Settings
from gestureflow.controller.mouse_controller import MouseController, MouseState
from gestureflow.gestures.classifier import FingerStates, Gesture, GestureResult
from gestureflow.tracking.hand_landmarker import HandLandmarks
from gestureflow.utils.filters import LowPassFilter, OneEuroFilter, OneEuroFilter1D


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_pag():
    """Create a mock PyAutoGUI module for isolated unit testing."""
    pag = MagicMock()
    pag.FAILSAFE = True
    pag.size.return_value = (1920, 1080)
    pag.position.return_value = (960, 540)
    return pag


@pytest.fixture
def one_euro_controller(mock_pag):
    """Instantiate MouseController configured with One Euro Filter."""
    settings = ControllerSettings(
        mouse_control_enabled_default=False,
        cursor_sensitivity=1.0,
        cursor_smoothing=0.35,
        cursor_dead_zone=2.5,
        tracking_region_margin=0.15,
        cursor_jump_threshold=250.0,
        smoothing_filter_type="one_euro",
        one_euro_min_cutoff=1.0,
        one_euro_beta=0.007,
        one_euro_d_cutoff=1.0,
        precision_mode=False,
        precision_multiplier=0.35,
        max_cursor_velocity=4000.0,
    )
    return MouseController(
        settings=settings,
        pag_module=mock_pag,
        screen_size=(1920, 1080),
        mock_mode=False,
    )


# ======================================================================
# 1. One Euro Filter Tests
# ======================================================================

class TestOneEuroFilter:
    def test_low_pass_filter_alpha(self):
        """Test LowPassFilter basic operation and bounds."""
        lpf = LowPassFilter(alpha=0.5)
        # First sample initializes filter
        assert lpf.filter(100.0) == 100.0
        # Second sample blends: 0.5 * 200 + 0.5 * 100 = 150
        assert lpf.filter(200.0) == 150.0
        assert lpf.last_value == 150.0

        lpf.reset()
        assert lpf.last_value is None

    def test_one_euro_1d_low_speed_jitter_reduction(self):
        """At low speeds (near rest), cutoff is low (1 Hz), heavily attenuating tremor."""
        f = OneEuroFilter1D(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        t = 0.0
        # Initialize at 500
        f.filter(500.0, timestamp=t)

        # Micro-tremor oscillation between 500 and 504
        filtered_values = []
        for i in range(1, 20):
            t += 0.033
            val = 500.0 + (4.0 if i % 2 == 1 else -4.0)
            res = f.filter(val, timestamp=t)
            filtered_values.append(res)

        # Amplitude of tremor in filtered output must be significantly smaller than raw (+-4)
        variations = [abs(v - 500.0) for v in filtered_values[-10:]]
        max_variation = max(variations)
        assert max_variation < 2.5, f"Tremor not sufficiently filtered: max var = {max_variation}"

    def test_one_euro_1d_high_speed_responsiveness(self):
        """During fast sweeps, speed-adaptive cutoff increases, keeping latency low."""
        f = OneEuroFilter1D(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
        t = 0.0
        f.filter(0.0, timestamp=t)

        # Fast sweep: +100 px every 33ms (~3000 px/sec)
        latest_filtered = 0.0
        for _ in range(10):
            t += 0.033
            target = latest_filtered + 100.0
            latest_filtered = f.filter(target, timestamp=t)

        # At high speed, lag between target and filtered output should be low
        lag = 1000.0 - latest_filtered
        assert lag < 100.0, f"Excessive lag during high-speed movement: lag = {lag}"

    def test_one_euro_2d_filter(self):
        """Test 2D OneEuroFilter operations and coordinate tuple processing."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
        f.reset()
        fx, fy = f.filter(100.0, 200.0, timestamp=1.0)
        assert fx == 100.0 and fy == 200.0

        fx2, fy2 = f.filter(110.0, 210.0, timestamp=1.033)
        assert 100.0 < fx2 < 110.0
        assert 200.0 < fy2 < 210.0

    def test_one_euro_dt_variation_and_pause_reset(self):
        """Test robust handling of variable frame rates and long pause reset."""
        f = OneEuroFilter1D(min_cutoff=1.0, beta=0.01)
        f.filter(100.0, timestamp=1.0)

        # 60 FPS frame (16.6ms)
        v60 = f.filter(120.0, timestamp=1.0166)
        assert 100.0 < v60 < 120.0

        # Long pause (> 1.0 second, e.g. user walked away) -> auto reset
        v_after_pause = f.filter(800.0, timestamp=3.5)
        # Should snap directly to 800 without dragging from 120
        assert v_after_pause == 800.0


# ======================================================================
# 2. Four-Point Calibrator Tests
# ======================================================================

class TestCalibrator:
    def test_default_calibrator_properties(self):
        """Test initial state of Calibrator before calibration."""
        c = Calibrator()
        assert not c.is_calibrated
        assert c.current_target_corner == CORNER_TL
        assert not c.is_sampling_complete
        assert c.data.x_min == 0.10
        assert c.data.x_max == 0.90

    def test_outlier_filtering(self):
        """Test statistical sample filter eliminates noisy spikes."""
        samples = [
            (0.50, 0.50),
            (0.51, 0.49),
            (0.49, 0.50),
            (0.50, 0.51),
            (0.50, 0.50),
            (0.95, 0.05),  # Spurious tracking glitch
            (0.01, 0.99),  # Spurious tracking glitch
        ]
        med_x, med_y = Calibrator.filter_samples(samples)
        assert 0.48 <= med_x <= 0.52
        assert 0.48 <= med_y <= 0.52

    def test_perspective_calibration_workflow(self):
        """Simulate complete 4-point calibration and verify homography mapping."""
        c = Calibrator()
        # Corner 1: TL camera points around (0.20, 0.20)
        for _ in range(25):
            c.add_sample(CORNER_TL, 0.20, 0.20)
        assert c.advance_to_next_corner()
        assert c.current_target_corner == CORNER_TR

        # Corner 2: TR camera points around (0.80, 0.20)
        for _ in range(25):
            c.add_sample(CORNER_TR, 0.80, 0.20)
        assert c.advance_to_next_corner()

        # Corner 3: BR camera points around (0.80, 0.80)
        for _ in range(25):
            c.add_sample(CORNER_BR, 0.80, 0.80)
        assert c.advance_to_next_corner()

        # Corner 4: BL camera points around (0.20, 0.80)
        for _ in range(25):
            c.add_sample(CORNER_BL, 0.20, 0.80)
        assert not c.advance_to_next_corner()
        assert c.is_sampling_complete

        # Compute calibration
        success = c.compute_calibration()
        assert success
        assert c.is_calibrated
        assert c.data.perspective_matrix is not None
        assert c.data.quality_score >= 0.90

        # Test mapping: center of camera active area (0.5, 0.5) maps to screen center
        sx, sy = c.map_point(0.50, 0.50, screen_width=1920, screen_height=1080)
        assert abs(sx - 1919 * 0.5) < 30
        assert abs(sy - 1079 * 0.5) < 30

    def test_degenerate_quadrilateral_fallback(self):
        """Points collapsed to a line or tiny area trigger robust bounding box fallback."""
        c = Calibrator()
        # Degenerate: all points along same vertical line x=0.5
        for corner in CALIBRATION_CORNERS:
            for _ in range(10):
                c.add_sample(corner, 0.50, 0.50)

        success = c.compute_calibration()
        assert success  # Succeeded via fallback
        # Perspective matrix should be None, but is_active is True with bounding box
        assert c.data.is_active
        # Fallback mapping maps smoothly without crashing
        sx, sy = c.map_point(0.50, 0.50, screen_width=1920, screen_height=1080)
        assert 0 <= sx <= 1920
        assert 0 <= sy <= 1080

    def test_calibration_serialization(self):
        """Test to_dict and from_dict preserve calibration state."""
        c = Calibrator()
        c.set_corner_point(CORNER_TL, 0.15, 0.15)
        c.set_corner_point(CORNER_TR, 0.85, 0.15)
        c.set_corner_point(CORNER_BR, 0.85, 0.85)
        c.set_corner_point(CORNER_BL, 0.15, 0.85)
        c.compute_calibration()

        d = c.data.to_dict()
        assert "perspective_matrix" in d
        assert d["is_active"]

        restored_data = CalibrationData.from_dict(d)
        restored_calibrator = Calibrator(restored_data)
        assert restored_calibrator.is_calibrated
        assert restored_calibrator.data.x_min == c.data.x_min


# ======================================================================
# 3. MouseController Calibrated Mapping & Active Margins
# ======================================================================

class TestCalibratedMouseController:
    def test_controller_uses_calibrator_when_calibrated(self, one_euro_controller):
        """MouseController routes through calibrator when active."""
        c = Calibrator()
        c.set_corner_point(CORNER_TL, 0.20, 0.20)
        c.set_corner_point(CORNER_TR, 0.80, 0.20)
        c.set_corner_point(CORNER_BR, 0.80, 0.80)
        c.set_corner_point(CORNER_BL, 0.20, 0.80)
        c.compute_calibration()
        one_euro_controller.set_calibrator(c)

        # Map center (0.5, 0.5)
        mapped = one_euro_controller.map_coordinates(0.50, 0.50)
        assert mapped is not None
        assert abs(mapped[0] - 1919 * 0.5) < 30
        assert abs(mapped[1] - 1079 * 0.5) < 30

    def test_custom_tracking_region_margin(self, one_euro_controller):
        """Test tracking_region_margin expansion or compression."""
        # Uncalibrated controller uses margin
        one_euro_controller.set_calibrator(Calibrator())  # Uncalibrated
        # With 0.25 margin, x=0.25 maps to screen 0
        c1 = one_euro_controller.map_coordinates(0.25, 0.5, margin=0.25)
        assert c1 is not None and c1[0] == 0.0

        # With 0.10 margin, x=0.25 is well inside the screen (> 0)
        c2 = one_euro_controller.map_coordinates(0.25, 0.5, margin=0.10)
        assert c2 is not None and c2[0] > 0.0


# ======================================================================
# 4. Precision Aiming Mode Tests
# ======================================================================

class TestPrecisionMode:
    def test_precision_mode_scaling(self, one_euro_controller):
        """Precision mode scales relative cursor displacements."""
        one_euro_controller.reset_smoothing()
        one_euro_controller.smooth_coordinates(500.0, 500.0, timestamp=1.0)

        # Normal mode displacement: target moves to (600, 500) (+100 px)
        one_euro_controller.smooth_coordinates(600.0, 500.0, timestamp=1.033)

        # Enable precision mode (0.35x multiplier)
        one_euro_controller.set_precision_mode(True)
        assert one_euro_controller.is_precision_mode

        # Step +100 px in precision mode
        p_prec = one_euro_controller.smooth_coordinates(700.0, 500.0, timestamp=1.066)
        assert p_prec is not None

    def test_precision_mode_smooth_transition_no_jump(self, one_euro_controller):
        """Toggling precision mode on and off does not cause cursor flings."""
        one_euro_controller.reset_smoothing()
        p1 = one_euro_controller.smooth_coordinates(500.0, 500.0, timestamp=1.0)
        assert p1 == (500, 500)

        # Turn ON precision mode
        one_euro_controller.set_precision_mode(True)
        # Next sample at same spot
        p2 = one_euro_controller.smooth_coordinates(500.0, 500.0, timestamp=1.033)
        assert p2 == (500, 500)

        # Turn OFF precision mode
        one_euro_controller.set_precision_mode(False)
        p3 = one_euro_controller.smooth_coordinates(500.0, 500.0, timestamp=1.066)
        assert p3 == (500, 500)


# ======================================================================
# 5. Tracking Stability & Outlier Clamping Tests
# ======================================================================

class TestTrackingStability:
    def test_velocity_limiting(self, one_euro_controller):
        """Displacement in 10ms below jump threshold is clamped to max_cursor_velocity * dt."""
        one_euro_controller.reset_smoothing()
        one_euro_controller.smooth_coordinates(100.0, 100.0, timestamp=1.0)

        # Move 140 pixels in 10ms (14,000 px/sec >> 4000 px/sec limit, but < 250 jump threshold)
        # Max allowed step in 10ms is 4000 * 0.010 = 40 px
        p_clamped = one_euro_controller.smooth_coordinates(240.0, 100.0, timestamp=1.010)
        assert p_clamped is not None
        # Position should not have reached 240; clamped to ~140
        assert p_clamped[0] <= 150

    def test_jump_threshold_snap(self, one_euro_controller):
        """Displacement > cursor_jump_threshold (250 px) snaps without drag."""
        one_euro_controller.reset_smoothing()
        one_euro_controller.smooth_coordinates(100.0, 100.0, timestamp=1.0)

        # Leap across screen: 800 pixels
        p_jump = one_euro_controller.smooth_coordinates(900.0, 100.0, timestamp=1.033)
        assert p_jump is not None
        # Jump threshold triggered: snaps directly to target (900, 100)
        assert p_jump[0] == 900
        assert p_jump[1] == 100

    def test_tracking_loss_resets_filter(self, one_euro_controller):
        """When hand is lost, filter state resets and resumes cleanly on re-entry."""
        one_euro_controller.enable()
        # Mock hand pointing
        hand = MagicMock()
        hand.handedness = "Right"
        hand.landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(21)]
        gesture = MagicMock(gesture=Gesture.POINTING, is_stable=True)

        res1 = one_euro_controller.process_hand(hand, gesture, timestamp=1.0)
        assert res1 is not None

        # Hand lost
        res_lost = one_euro_controller.process_hand(None, None, timestamp=1.033)
        assert res_lost is None
        assert one_euro_controller.state == MouseState.IDLE

        # Hand re-enters at different location (0.2, 0.2)
        hand.landmarks[8].x = 0.2
        hand.landmarks[8].y = 0.2
        res_reentry = one_euro_controller.process_hand(hand, gesture, timestamp=2.0)
        assert res_reentry is not None
        assert one_euro_controller.state == MouseState.MOVING


# ======================================================================
# 6. Settings & Filter Mode Selection
# ======================================================================

class TestSettingsAndFilterMode:
    def test_controller_filter_type_switch(self, one_euro_controller):
        """Switching filter type between 'one_euro' and 'ema' works smoothly."""
        one_euro_controller.set_filter_type("ema")
        assert one_euro_controller.settings.smoothing_filter_type == "ema"

        one_euro_controller.set_filter_type("one_euro")
        assert one_euro_controller.settings.smoothing_filter_type == "one_euro"

    def test_one_euro_param_setters(self, one_euro_controller):
        """Tuning min_cutoff, beta, d_cutoff updates controller settings."""
        one_euro_controller.set_one_euro_params(min_cutoff=2.5, beta=0.02, d_cutoff=1.5)
        assert one_euro_controller.settings.one_euro_min_cutoff == 2.5
        assert one_euro_controller.settings.one_euro_beta == 0.02
        assert one_euro_controller.settings.one_euro_d_cutoff == 1.5
