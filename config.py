"""Training configuration for adversarial LED pattern optimization.

Every experiment parameter is defined here. The training entry point
(``train.py``) dumps this class verbatim into the console and into the header
of each run log, so a log file is sufficient to reproduce a run.
"""


class TrainingConfig:

    # =========================================================================
    # NETWORK
    # =========================================================================
    SERVER_HOST = '127.0.0.1'
    SERVER_PORT = 5000

    # =========================================================================
    # DETECTOR CAMERA (attached to the training host)
    # =========================================================================
    PC_CAMERA_INDEX    = 0
    CAMERA_WARMUP_TIME = 2.0   # seconds

    # =========================================================================
    # DETECTOR
    # =========================================================================
    YOLO_MODEL   = 'yolov8n.pt'
    TARGET_CLASS = 0    # COCO class index for "person"
    # Attack success rate is reported offline, over the recorded evaluation
    # frames, as the fraction with no person detection matching the ground
    # truth at IoU >= 0.5 and confidence >= 0.6. Training uses the dense
    # confidence signal below instead, because a binary per-frame signal is
    # too sparse for reinforcement learning in a physical loop.

    # =========================================================================
    # LED ARRAY / ACTION SPACE
    # =========================================================================
    BOARD_H  = 16   # LED rows per panel
    BOARD_W  = 16   # LED columns per panel
    N_BOARDS = 39   # number of panels
    # Total LEDs: BOARD_H * BOARD_W * N_BOARDS = 9984
    # Direct control dimension: 3 * 9984 = 29952
    # Wiring is serpentine: even rows run left to right, odd rows right to left.

    # Superpixel size: one RGB value drives a SUPERPIXEL_SIZE x SUPERPIXEL_SIZE
    # block of LEDs. Must divide both BOARD_H and BOARD_W exactly; admissible
    # values are 1, 2, 4, 8 and 16, giving compression factors of 1x, 4x, 16x,
    # 64x and 256x respectively.
    SUPERPIXEL_SIZE = 4

    MAX_BRIGHTNESS = 10   # per-channel LED brightness ceiling (0-255)

    # =========================================================================
    # STATE
    # Per-channel mean and standard deviation of the two onboard cameras,
    # so state_dim = 12 (3 channels x 2 statistics x 2 cameras).
    # =========================================================================
    IMAGE_SIZE = 64   # images are resized to this before statistics are taken

    # =========================================================================
    # EXPLORATION NOISE
    # =========================================================================
    # Added to the state BEFORE it reaches the actor (broadens exploration).
    STATE_NOISE_STD  = 0.05
    # Added to the action AFTER the actor emits it (fine-grained perturbation).
    ACTION_NOISE_STD = 0.02

    # =========================================================================
    # REWARD FUNCTION
    # reward = REWARD_A * (DETECTION_THRESHOLD - detection_confidence)
    #        + REWARD_B * action_variance
    #        + REWARD_C * squared_difference_from_previous_action
    # =========================================================================
    DETECTION_THRESHOLD  = 0.9   # detection confidence the attack is scored against
    REWARD_A             = 1e1   # weight of the detection term
    REWARD_B             = 1e2   # weight of the variance term (rewards diversity)
    REWARD_C             = 1e1   # weight of the temporal term (rewards change)
    DETECTION_AVG_FRAMES = 3     # frames averaged per step to denoise the reward

    # =========================================================================
    # SAC HYPERPARAMETERS
    # =========================================================================
    GAMMA    = 0.99
    TAU      = 0.005
    ACTOR_LR  = 1e-4
    CRITIC_LR = 1e-4
    ALPHA_LR  = 1e-4

    HIDDEN_DIMS    = [256, 256]
    GRAD_CLIP_NORM = 1.0    # maximum gradient norm; None disables clipping

    BATCH_SIZE      = 64
    BUFFER_SIZE     = 50_000
    LEARNING_STARTS = 10   # begin updating after this many stored transitions

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    NUM_EPISODES          = 2000   # ~10-12 s per episode, ~5-6 h per run
    MAX_STEPS_PER_EPISODE = 1
    DEVICE                = 'cuda'
    VERBOSE               = True

    # =========================================================================
    # OUTPUT
    # =========================================================================
    MODELS_DIR         = './checkpoints'
    LOGS_DIR           = './logs'
    DATASET_DIR        = './dataset'
    # Checkpoint often enough that the deployed policy can be selected by
    # minimum mean confidence on a held-out sequence, as in the paper.
    SAVE_MODEL_EVERY   = 100
    SAVE_DATASET       = False
    SAVE_VISUALIZATION = False

    # =========================================================================
    # SOCKET TIMEOUTS
    # =========================================================================
    SOCKET_TIMEOUT         = 10   # seconds to wait for onboard camera frames
    PATTERN_RENDER_TIMEOUT = 60   # seconds to wait for the render acknowledgement

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    SHOW_YOLO_DETECTIONS = True
    SHOW_LED_PATTERN     = True
    SHOW_ENV_IMAGES      = True
    VISUALIZATION_DELAY  = 1   # ms; 1 refreshes every frame, 0 waits for a keypress

    WINDOW_POSITIONS = {
        'Detector View': (0,   0),
        'LED Pattern':   (700, 0),
        'Onboard Camera (front)': (0,   550),
        'Onboard Camera (back)':  (700, 550),
    }
