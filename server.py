"""LED controller service, run on the single-board computer worn with the garment.

Per control step the service:

1. captures the two onboard camera frames and sends them to the training host;
2. receives the full per-LED pattern from the training host;
3. renders the pattern on the LED matrix;
4. acknowledges the render so the host can score the resulting scene.

Both GPIO rendering and camera capture are stubbed out, because the driver is
specific to the LED panels and the camera modules in use. With ``SIMULATE``
left enabled the service runs the complete protocol against synthetic frames,
which is enough to exercise the training loop end to end on a single machine.

Run on the controller (or locally, for a protocol smoke test):

    python server.py
"""

import socket
import time

import numpy as np

from utils.socket_utils import recv_data, send_data


# =============================================================================
# Configuration
# =============================================================================
class ServerConfig:
    HOST = '0.0.0.0'
    PORT = 5000

    #: Run against synthetic frames and a simulated LED driver.
    SIMULATE = True

    #: Resolution of the synthetic onboard frames.
    FRAME_HEIGHT = 720
    FRAME_WIDTH  = 720

    #: Wall-clock time the simulated LED driver spends on one render.
    SIMULATED_RENDER_TIME = 0.5   # seconds

    #: Seconds to wait for a pattern from the training host.
    PATTERN_TIMEOUT = 60


# =============================================================================
# LED driver
# =============================================================================
class LEDController:
    """Drives the LED matrix, or simulates doing so."""

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        print(f"[INFO] LED controller mode: {'simulation' if simulate else 'GPIO'}")

    def render_pattern(self, pattern: np.ndarray, render_time: float = 0.5) -> bool:
        """Write one full RGB pattern to the matrix. Returns True on success."""
        if not self.simulate:
            raise NotImplementedError(
                "GPIO rendering is hardware specific and is not part of this "
                "release. Implement it here for your LED panels, or leave "
                "ServerConfig.SIMULATE enabled."
            )

        print(f"[INFO] Rendering pattern: size={len(pattern)}  "
              f"min={pattern.min()}  max={pattern.max()}  "
              f"mean={pattern.mean():.1f}")
        time.sleep(render_time)
        print("[INFO] Pattern rendered.")
        return True


# =============================================================================
# Onboard cameras
# =============================================================================
class OnboardCameras:
    """Reads the forward- and backward-facing observation cameras.

    In simulation mode it returns synthetic frames whose brightness and colour
    balance drift from step to step, so the state vector the training host
    derives from them is not constant.
    """

    def __init__(self, simulate: bool = True, seed: int = 0,
                 height: int = ServerConfig.FRAME_HEIGHT,
                 width: int = ServerConfig.FRAME_WIDTH):
        self.simulate = simulate
        self.height   = height
        self.width    = width
        self._rng     = np.random.default_rng(seed)

        print(f"[INFO] Onboard cameras mode: "
              f"{'simulation' if simulate else 'hardware'}")

    def capture(self):
        """Return ``(front, back)`` BGR frames, or ``(None, None)`` on failure."""
        if not self.simulate:
            raise NotImplementedError(
                "Camera capture depends on the camera modules in use and is "
                "not part of this release. Implement it here, or leave "
                "ServerConfig.SIMULATE enabled."
            )
        return self._synthetic_frame(), self._synthetic_frame()

    def _synthetic_frame(self) -> np.ndarray:
        """Build a frame with a random colour cast and a vertical gradient."""
        gradient = np.linspace(0.35, 0.95, self.height, dtype=np.float32)
        base     = np.repeat(gradient[:, None], self.width, axis=1)

        cast  = self._rng.uniform(0.55, 1.0, size=3).astype(np.float32)
        frame = base[:, :, None] * cast[None, None, :]
        frame += self._rng.normal(0.0, 0.04, size=frame.shape).astype(np.float32)

        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


# =============================================================================
# Service
# =============================================================================
def run_server():
    print("=" * 70)
    print("LED CONTROLLER SERVICE")
    print("=" * 70)

    led_controller = LEDController(simulate=ServerConfig.SIMULATE)
    cameras        = OnboardCameras(simulate=ServerConfig.SIMULATE)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        server_socket.bind((ServerConfig.HOST, ServerConfig.PORT))
        server_socket.listen(1)

        print(f"[INFO] Listening on {ServerConfig.HOST}:{ServerConfig.PORT}")
        print("[INFO] Waiting for the training host...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"[INFO] Training host connected: {addr}")
            step = 0

            try:
                while True:
                    step += 1
                    print(f"\n[Step {step}] " + "-" * 50)

                    # 1. Capture the onboard frames.
                    front, back = cameras.capture()
                    if front is None:
                        print("[ERROR] Camera capture failed. Stopping.")
                        break
                    print(f"[INFO] Frames captured: {front.shape}, {back.shape}")

                    # 2. Send them to the training host.
                    if not send_data(conn, (front, back)):
                        print("[ERROR] Could not send the frames.")
                        break
                    print("[INFO] Frames sent.")

                    # 3. Receive the LED pattern.
                    pattern = recv_data(conn, timeout=ServerConfig.PATTERN_TIMEOUT)
                    if pattern is None:
                        print("[ERROR] No pattern received from the training host.")
                        break
                    pattern = np.asarray(pattern)
                    print(f"[INFO] Pattern received: shape={pattern.shape}  "
                          f"min={pattern.min()}  max={pattern.max()}")

                    # 4. Render it.
                    rendered = led_controller.render_pattern(
                        pattern, render_time=ServerConfig.SIMULATED_RENDER_TIME
                    )

                    # 5. Acknowledge.
                    acknowledgement = {
                        'rendered':  rendered,
                        'step':      step,
                        'timestamp': time.time(),
                    }
                    if not send_data(conn, acknowledgement):
                        print("[ERROR] Could not send the acknowledgement.")
                        break
                    print(f"[INFO] Step {step} complete.")

            except KeyboardInterrupt:
                print("\n[INFO] Stopped by the operator.")
            except Exception as exc:
                print(f"[ERROR] Unhandled exception: {exc}")
            finally:
                print("[INFO] Closing the connection.")


if __name__ == '__main__':
    run_server()
