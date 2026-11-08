"""Closed-loop physical environment for adversarial LED pattern optimization.

State
    Per-channel mean and standard deviation of the two onboard camera frames
    (12 values in total).

Action
    A superpixel LED pattern. Each value drives a ``SUPERPIXEL_SIZE**2`` block
    of diodes, which keeps the action dimensionality tractable for off-policy
    reinforcement learning.

LED layout
    ``N_BOARDS`` panels of ``BOARD_H x BOARD_W`` diodes, wired as a serpentine:
    even rows run left to right (columns 0..W-1), odd rows run right to left
    (columns W-1..0).

Superpixel mapping
    Each panel is divided into ``(H/S) x (W/S)`` blocks of ``S x S`` diodes and
    one RGB triplet is assigned to every diode in a block. The expansion
    accounts for the serpentine wiring, so blocks stay spatially contiguous on
    the garment.
"""

import socket
import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils.socket_utils import recv_data, send_data
from utils.visualization import Visualizer

#: Reward returned when the physical loop fails and the episode is aborted.
FAILURE_REWARD = -10.0


class AdversarialJacketEnv:
    """Reinforcement learning environment backed by the physical LED garment."""

    def __init__(self, cfg):
        self.cfg         = cfg
        self.socket      = None
        self.pc_camera   = None
        self.last_visualization = None
        self.prev_action = None

        self.board_h         = cfg.BOARD_H
        self.board_w         = cfg.BOARD_W
        self.n_boards        = cfg.N_BOARDS
        self.n_leds          = cfg.BOARD_H * cfg.BOARD_W * cfg.N_BOARDS
        self.superpixel_size = cfg.SUPERPIXEL_SIZE
        self.max_action      = cfg.MAX_BRIGHTNESS / 255.0

        if cfg.BOARD_H % cfg.SUPERPIXEL_SIZE != 0:
            raise ValueError(
                f"SUPERPIXEL_SIZE ({cfg.SUPERPIXEL_SIZE}) must divide "
                f"BOARD_H ({cfg.BOARD_H}) exactly."
            )
        if cfg.BOARD_W % cfg.SUPERPIXEL_SIZE != 0:
            raise ValueError(
                f"SUPERPIXEL_SIZE ({cfg.SUPERPIXEL_SIZE}) must divide "
                f"BOARD_W ({cfg.BOARD_W}) exactly."
            )

        side = cfg.SUPERPIXEL_SIZE
        sp_h = cfg.BOARD_H // side
        sp_w = cfg.BOARD_W // side
        self.n_superpixels = cfg.N_BOARDS * sp_h * sp_w
        self.action_dim    = self.n_superpixels * 3
        self.state_dim     = 12   # per-channel mean and std for two cameras

        # Precompute the superpixel -> LED index mapping once.
        self.led_map = self._build_led_map()

        self.vis = Visualizer(
            window_positions=cfg.WINDOW_POSITIONS,
            delay=cfg.VISUALIZATION_DELAY,
        )

        print("[INFO] Loading the detector...")
        self.yolo = YOLO(cfg.YOLO_MODEL)
        print("[INFO] Detector loaded.")

        print("[INFO] Opening the detector camera...")
        self.pc_camera = cv2.VideoCapture(cfg.PC_CAMERA_INDEX)
        if not self.pc_camera.isOpened():
            raise RuntimeError(
                f"Could not open the detector camera at index {cfg.PC_CAMERA_INDEX}."
            )
        print(f"[INFO] Warming the camera up ({cfg.CAMERA_WARMUP_TIME}s)...")
        time.sleep(cfg.CAMERA_WARMUP_TIME)
        for _ in range(5):
            self.pc_camera.read()
        print("[INFO] Detector camera ready.")

        print(f"[INFO] LED layout: {self.n_boards} panels x {self.board_h}x{self.board_w} "
              f"= {self.n_leds} diodes")
        print(f"[INFO] Superpixel {side}x{side} -> {self.n_superpixels} superpixels, "
              f"action_dim={self.action_dim} "
              f"(compression {self.n_leds * 3 / self.action_dim:.1f}x)")

    # =========================================================================
    # LED mapping
    # =========================================================================
    def _build_led_map(self) -> list:
        """Map each superpixel index to the absolute LED indices it drives."""
        side = self.superpixel_size
        height = self.board_h
        width  = self.board_w
        sp_h = height // side
        sp_w = width // side

        def snake_led(board: int, row: int, col: int) -> int:
            local_col = col if row % 2 == 0 else (width - 1 - col)
            return board * height * width + row * width + local_col

        led_map = []
        for board in range(self.n_boards):
            for sp_row in range(sp_h):
                for sp_col in range(sp_w):
                    leds = []
                    for r in range(sp_row * side, (sp_row + 1) * side):
                        for c in range(sp_col * side, (sp_col + 1) * side):
                            leds.append(snake_led(board, r, c))
                    led_map.append(leds)

        return led_map

    def _expand_pattern(self, action: np.ndarray) -> np.ndarray:
        """Expand a superpixel action into a full per-LED pattern.

        Args:
            action: array of shape ``[n_superpixels * 3]`` with values in
                ``[0, max_action]``.

        Returns:
            ``uint8`` array of shape ``[n_leds * 3]`` with values in ``[0, 255]``.
        """
        full   = np.zeros(self.n_leds * 3, dtype=np.uint8)
        colors = (action.reshape(-1, 3) * 255).astype(np.uint8)

        for sp_idx, led_indices in enumerate(self.led_map):
            color = colors[sp_idx]
            for led_idx in led_indices:
                full[led_idx * 3: led_idx * 3 + 3] = color

        return full

    # =========================================================================
    # Connection to the LED controller
    # =========================================================================
    def connect_to_controller(self) -> bool:
        """Open a TCP connection to the LED controller."""
        print(f"[INFO] Connecting to {self.cfg.SERVER_HOST}:{self.cfg.SERVER_PORT}...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        try:
            self.socket.connect((self.cfg.SERVER_HOST, self.cfg.SERVER_PORT))
            print("[INFO] Connected to the LED controller.")
            return True
        except Exception as exc:
            print(f"[ERROR] Connection failed: {exc}")
            return False

    def ensure_connection(self) -> bool:
        """Return True if the controller is reachable, reconnecting if needed."""
        if self.socket is None:
            return self.connect_to_controller()
        try:
            self.socket.settimeout(0.1)
            self.socket.send(b'')
            self.socket.settimeout(None)
            return True
        except (socket.error, socket.timeout):
            print("[WARNING] Connection lost. Reconnecting...")
            self.socket.close()
            self.socket = None
            return self.connect_to_controller()

    # =========================================================================
    # State
    # =========================================================================
    def get_state(self):
        """Read both onboard frames and reduce them to the state vector.

        Returns:
            ``(state, (front, back))`` where ``state`` has shape ``[12]``, or
            ``None`` if the frames could not be received.
        """
        images = recv_data(self.socket, timeout=self.cfg.SOCKET_TIMEOUT)
        if images is None:
            print("[ERROR] No onboard camera frames received from the controller.")
            self.socket.close()
            self.socket = None
            return None

        img_front, img_back = images

        if self.cfg.SHOW_ENV_IMAGES:
            self.vis.show_onboard_images(img_front, img_back)

        def image_stats(img):
            resized = cv2.resize(img, (self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE))
            norm    = resized.astype(np.float32) / 255.0
            return np.concatenate([
                norm.mean(axis=(0, 1)),
                norm.std(axis=(0, 1)),
            ])

        state = np.concatenate([image_stats(img_front), image_stats(img_back)])
        return state, (img_front, img_back)

    # =========================================================================
    # Pattern transmission
    # =========================================================================
    def send_pattern_and_wait(self, action: np.ndarray) -> bool:
        """Expand, transmit and wait for the controller to confirm the render."""
        pattern_full = self._expand_pattern(action)
        if not send_data(self.socket, pattern_full):
            return False
        response = recv_data(self.socket, timeout=self.cfg.PATTERN_RENDER_TIMEOUT)
        return response is not None and response.get('rendered', False)

    # =========================================================================
    # Detection
    # =========================================================================
    def capture_and_detect(self):
        """Capture a detector frame and run person detection on it.

        Returns:
            ``(max_confidence, frame)``, or ``(None, None)`` if the frame could
            not be captured.
        """
        ret, frame = self.pc_camera.read()
        if not ret:
            print("[ERROR] Could not capture a frame from the detector camera.")
            return None, None

        results = self.yolo(frame, verbose=False)

        person_confidences = []
        person_boxes       = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                mask  = boxes.cls == self.cfg.TARGET_CLASS
                if mask.sum() > 0:
                    person_confidences.extend(boxes.conf[mask].cpu().numpy().tolist())
                    person_boxes.extend(boxes.xyxy[mask].cpu().numpy().tolist())

        max_confidence = max(person_confidences) if person_confidences else 0.0

        if self.cfg.SHOW_YOLO_DETECTIONS:
            vis_frame = self.vis.show_detections(
                frame, person_boxes, person_confidences, max_confidence
            )
            if self.cfg.SAVE_VISUALIZATION:
                self.last_visualization = vis_frame

        return max_confidence, frame

    # =========================================================================
    # Reinforcement learning interface
    # =========================================================================
    def step(self, action: np.ndarray):
        """Render one action on the garment and score the resulting frame.

        Returns:
            ``(reward, done, info)``. On a hardware or detector failure the
            episode is aborted with ``(FAILURE_REWARD, True, {})``.
        """
        if self.cfg.SHOW_LED_PATTERN:
            self.vis.show_led_pattern(action, self.n_superpixels,
                                      self.cfg.MAX_BRIGHTNESS, self.superpixel_size)

        if not self.send_pattern_and_wait(action):
            print("[ERROR] Pattern transmission or render acknowledgement failed.")
            return FAILURE_REWARD, True, {}

        # Average the detector confidence over several frames to damp the
        # measurement noise that enters the reward.
        conf_values = []
        last_frame  = None
        for _ in range(self.cfg.DETECTION_AVG_FRAMES):
            conf, frame = self.capture_and_detect()
            if conf is None:
                print("[ERROR] Detection failed.")
                return FAILURE_REWARD, True, {}
            conf_values.append(conf)
            last_frame = frame

        detection_conf = float(np.mean(conf_values))

        # Reward terms.
        reward_det  = self.cfg.DETECTION_THRESHOLD - detection_conf
        reward_var  = float(np.var(action))
        reward_diff = float(np.mean((action - self.prev_action) ** 2)) \
            if self.prev_action is not None else 0.0

        reward = (self.cfg.REWARD_A * reward_det
                  + self.cfg.REWARD_B * reward_var
                  + self.cfg.REWARD_C * reward_diff)

        if self.cfg.VERBOSE:
            print(f"  [reward] detection={self.cfg.REWARD_A * reward_det:.3f}  "
                  f"variance={self.cfg.REWARD_B * reward_var:.3f}  "
                  f"temporal={self.cfg.REWARD_C * reward_diff:.3f}  "
                  f"total={reward:.3f}  confidence={detection_conf:.3f}")

        self.prev_action = action.copy()

        info = {'detection_confidence': detection_conf, 'frame': last_frame}
        return reward, False, info

    def reset(self):
        """Reconnect if needed, clear the previous action and read the state.

        Returns:
            ``(state, raw_images)``, or ``(None, None)`` on failure.
        """
        self.prev_action = None
        if not self.ensure_connection():
            print("[ERROR] Could not reach the LED controller.")
            return None, None
        result = self.get_state()
        if result is None:
            return None, None
        return result

    def close(self):
        """Release the socket, the camera and every visualization window."""
        if self.socket:
            self.socket.close()
        if self.pc_camera:
            self.pc_camera.release()
        self.vis.destroy_all()
