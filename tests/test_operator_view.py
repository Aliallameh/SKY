from __future__ import annotations

import numpy as np
import pytest

from skyscouter.output.operator_view import LiveOperatorView


def test_operator_view_letterboxes_display_frame_to_fixed_resolution():
    view = LiveOperatorView(
        mode="mjpeg",
        port=0,
        display_width=1920,
        display_height=1080,
        fullscreen=True,
    )
    try:
        image = np.full((480, 640, 3), (20, 120, 220), dtype=np.uint8)
        display = view._resize_for_display(image)
    finally:
        view.close()

    assert display.shape == (1080, 1920, 3)
    assert np.all(display[:, :240] == 0)
    assert np.all(display[:, 1680:] == 0)
    assert np.any(display[:, 240:1680] != 0)


def test_operator_view_from_config_reads_hdmi_display_options():
    view = LiveOperatorView.from_config(
        {
            "mode": "mjpeg",
            "port": 0,
            "fullscreen": True,
            "display_width": 1920,
            "display_height": 1080,
            "window_x": 0,
            "window_y": 0,
            "window_backend": "gstreamer",
            "display_fps": 5,
        }
    )
    try:
        assert view.fullscreen is True
        assert view.display_width == 1920
        assert view.display_height == 1080
        assert view.window_x == 0
        assert view.window_y == 0
        assert view.window_backend == "gstreamer"
        assert view.display_fps == 5
    finally:
        view.close()


def test_operator_view_rejects_invalid_display_size():
    with pytest.raises(ValueError, match="display_width"):
        LiveOperatorView(mode="mjpeg", display_width=0)


def test_operator_view_builds_gstreamer_display_command():
    view = LiveOperatorView(
        mode="mjpeg",
        port=0,
        window_backend="gstreamer",
        window_x=0,
        window_y=0,
        display_fps=10,
    )
    try:
        cmd = view._build_gstreamer_command(1920, 1080)
    finally:
        view.close()

    assert cmd[:4] == ["gst-launch-1.0", "-q", "fdsrc", "fd=0"]
    assert "videoparse" in cmd
    assert "format=bgr" in cmd
    assert "width=1920" in cmd
    assert "height=1080" in cmd
    assert "framerate=10/1" in cmd
    assert "nveglglessink" in cmd
    assert "window-width=1920" in cmd
    assert "window-height=1080" in cmd
