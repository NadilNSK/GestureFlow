"""
Tests for gestureflow.utils.helpers
"""

from __future__ import annotations

import time

import pytest

from gestureflow.utils.helpers import FpsCounter, clamp, lerp, retry


class TestFpsCounter:
    def test_initial_fps_is_zero(self):
        c = FpsCounter()
        assert c.fps == 0.0

    def test_fps_increases_with_ticks(self):
        c = FpsCounter(window=10)
        for _ in range(10):
            c.tick()
            time.sleep(0.001)
        assert c.fps > 0

    def test_window_capped(self):
        c = FpsCounter(window=5)
        for _ in range(20):
            c.tick()
        # Only last 5 timestamps are kept.
        assert len(c._timestamps) == 5


class TestClamp:
    def test_below_lo(self):
        assert clamp(-5.0, 0.0, 1.0) == 0.0

    def test_above_hi(self):
        assert clamp(2.0, 0.0, 1.0) == 1.0

    def test_in_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5


class TestLerp:
    def test_t_zero(self):
        assert lerp(0.0, 10.0, 0.0) == pytest.approx(0.0)

    def test_t_one(self):
        assert lerp(0.0, 10.0, 1.0) == pytest.approx(10.0)

    def test_midpoint(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)


class TestRetry:
    def test_succeeds_first_try(self):
        result = retry(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        result = retry(flaky, attempts=3, delay=0)
        assert result == "ok"
        assert calls["n"] == 3

    def test_raises_after_max_attempts(self):
        with pytest.raises(RuntimeError, match="All 2 attempts failed"):
            retry(lambda: (_ for _ in ()).throw(ValueError("fail")), attempts=2, delay=0)
