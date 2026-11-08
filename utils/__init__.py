"""Replay buffer, TCP transport helpers and live visualization.

Submodules are imported explicitly (``from utils.socket_utils import ...``) so
that the LED controller, which runs headless on the single-board computer, can
use the transport helpers without pulling in the OpenCV-based visualizer.
"""
