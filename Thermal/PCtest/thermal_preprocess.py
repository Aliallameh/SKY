"""
thermal_preprocess.py

Preprocess live thermal camera frames before detection.

Main job:
    - Optionally rotate the raw frame first
    - Split the two-view thermal image
    - Return only selected view
"""


def _get_config_value(config, key, default):
    if config is None:
        return default

    if isinstance(config, dict):
        return config.get(key, default)

    return getattr(config, key, default)


def _rotate_frame(frame, rotate):
    """
    Rotate frame before splitting.

    rotate options:
        "none"
        "clockwise"
        "counterclockwise"
        "180"
    """
    import cv2

    if rotate == "none":
        return frame

    if rotate == "clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if rotate == "counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)

    raise ValueError(
        f"Invalid rotate='{rotate}'. "
        "Use 'none', 'clockwise', 'counterclockwise', or '180'."
    )


def preprocess_frame(frame, preprocess_config=None):
    if frame is None:
        raise ValueError("Input frame is None.")

    enabled = _get_config_value(preprocess_config, "enabled", True)

    if not enabled:
        return frame

    # ------------------------------------------------------------
    # Important:
    # Rotate BEFORE splitting.
    # ------------------------------------------------------------
    rotate = _get_config_value(
        preprocess_config,
        "rotate",
        "none",
    )

    frame = _rotate_frame(frame, rotate)

    split_mode = _get_config_value(
        preprocess_config,
        "split_mode",
        "horizontal",
    )

    selected_view = _get_config_value(
        preprocess_config,
        "selected_view",
        "top",
    )

    height, width = frame.shape[:2]

    if split_mode == "horizontal":
        split_y = _get_config_value(preprocess_config, "split_y", None)

        if split_y is None:
            split_y = height // 2

        split_y = int(split_y)

        if split_y <= 0 or split_y >= height:
            raise ValueError(
                f"Invalid split_y={split_y}. Frame height is {height}."
            )

        top_frame = frame[:split_y, :]
        bottom_frame = frame[split_y:, :]

        if selected_view == "top":
            return top_frame

        if selected_view == "bottom":
            return bottom_frame

        if selected_view == "full":
            return frame

        raise ValueError(
            f"Invalid selected_view='{selected_view}' for horizontal split. "
            "Use 'top', 'bottom', or 'full'."
        )

    elif split_mode == "vertical":
        split_x = _get_config_value(preprocess_config, "split_x", None)

        if split_x is None:
            split_x = width // 2

        split_x = int(split_x)

        if split_x <= 0 or split_x >= width:
            raise ValueError(
                f"Invalid split_x={split_x}. Frame width is {width}."
            )

        left_frame = frame[:, :split_x]
        right_frame = frame[:, split_x:]

        if selected_view == "left":
            return left_frame

        if selected_view == "right":
            return right_frame

        if selected_view == "full":
            return frame

        raise ValueError(
            f"Invalid selected_view='{selected_view}' for vertical split. "
            "Use 'left', 'right', or 'full'."
        )

    elif split_mode == "none":
        return frame

    else:
        raise ValueError(
            f"Invalid split_mode='{split_mode}'. "
            "Use 'horizontal', 'vertical', or 'none'."
        )