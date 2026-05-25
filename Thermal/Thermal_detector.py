"""
thermal_detector.py

Very simple thermal detector.

Purpose:
    Algorithm-only module.

Input:
    One OpenCV frame/image.

Output:
    List of Box objects.


This file should NOT handle:
    - video reading
    - live display
    - multithreading
    - video recording

Use from another script:

    from thermal_detector import detect_frame

    boxes = detect_frame(frame, algorithm_config)
"""
