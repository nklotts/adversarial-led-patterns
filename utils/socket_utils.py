"""Length-prefixed message passing over TCP.

Wire format: a 4-byte big-endian payload length followed by the pickled
payload. Both ends of the control loop (the training host and the LED
controller) use these helpers, so the format is defined in one place only.

Note:
    ``pickle`` executes arbitrary code on load. This link is intended for a
    trusted point-to-point connection on the local experiment network; do not
    expose the controller port to an untrusted network.
"""

import pickle
import socket
import struct

MAX_PAYLOAD_BYTES = 500 * 1024 * 1024


def send_data(sock: socket.socket, data) -> bool:
    """Serialize ``data`` and write it to ``sock``. Returns True on success."""
    try:
        serialized = pickle.dumps(data)
        size       = struct.pack('>I', len(serialized))
        sock.sendall(size + serialized)
        return True
    except Exception as exc:
        print(f"[ERROR] send_data: {exc}")
        return False


def recv_data(sock: socket.socket, timeout: float = None):
    """Read one length-prefixed message from ``sock`` and deserialize it.

    Args:
        sock: a connected socket.
        timeout: read timeout in seconds; ``None`` blocks indefinitely.

    Returns:
        The deserialized object, or ``None`` on timeout, disconnection or a
        malformed frame.
    """
    if timeout is not None:
        sock.settimeout(timeout)

    try:
        raw_size = b''
        while len(raw_size) < 4:
            chunk = sock.recv(4 - len(raw_size))
            if not chunk:
                return None
            raw_size += chunk

        data_size = struct.unpack('>I', raw_size)[0]

        if data_size > MAX_PAYLOAD_BYTES:
            print(f"[ERROR] recv_data: payload too large ({data_size} bytes)")
            return None

        data = b''
        while len(data) < data_size:
            chunk = sock.recv(min(65536, data_size - len(data)))
            if not chunk:
                return None
            data += chunk

        return pickle.loads(data)

    except socket.timeout:
        print("[ERROR] recv_data: timed out")
        return None
    except Exception as exc:
        print(f"[ERROR] recv_data: {exc}")
        return None
    finally:
        if timeout is not None:
            sock.settimeout(None)
