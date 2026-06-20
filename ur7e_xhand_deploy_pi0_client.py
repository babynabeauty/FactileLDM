#!/usr/bin/env python3
"""
Deploy a PI0 policy server on the UR7e + XHand robot.

This client runs on the robot machine. It reads UR7e + XHand observations,
sends state/images to a remote PI0 inference server over websocket, receives
an action chunk, and executes it on the robot.

# 1. 不带触觉的 pi0 baseline
--policy-input-mode vanilla

# 2. 当前 tactile observation 的 obs-AE
--policy-input-mode obs_ae

# 3. structured single-AE，10帧历史 tactile
--policy-input-mode structured_single_ae

"""

import argparse
import collections
import functools
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_DATASET_ROOT = REPO_ROOT / "data"
DEFAULT_DATASET_NAME = "grasp_pipette"
DEFAULT_TASK = "pick up the pipette"
DEFAULT_SERVER_PORT = 8990
STRUCTURED_TACTILE_HISTORY_OFFSETS = (-18, -16, -14, -12, -10, -8, -6, -4, -2, 0)
TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_BLOCK_START = 52
TACTILE_CALC_FORCE_OFFSET = 0
TACTILE_AXES = ("x", "y", "z")

FALLBACK_STATE_NAMES = [
    *[f"arm_joint_{i}.pos" for i in range(6)],
    *[f"arm_joint_{i}.vel" for i in range(6)],
    *[f"arm_ee_pose.{i:02d}" for i in range(16)],
    *[f"hand_joint_{i}.pos" for i in range(12)],
    *[f"hand_joint_{i}.torque" for i in range(12)],
]

FALLBACK_ACTION_NAMES = [
    *[f"arm_joint_{i}.pos" for i in range(6)],
    *[f"hand_joint_{i}.pos" for i in range(12)],
]


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


def import_msgpack():
    try:
        return importlib.import_module("msgpack")
    except ImportError as exc:
        raise ImportError(
            "Missing Python package 'msgpack'. Install it in the deployment environment, e.g. "
            "`pip install msgpack websockets`."
        ) from exc


def import_websockets_sync_client():
    try:
        return importlib.import_module("websockets.sync.client")
    except ImportError as exc:
        raise ImportError(
            "Missing Python package 'websockets'. Install it in the deployment environment, e.g. "
            "`pip install msgpack websockets`."
        ) from exc


def init_logging() -> None:
    try:
        lerobot_utils = importlib.import_module("lerobot.utils.utils")
        lerobot_utils.init_logging()
    except (ImportError, AttributeError):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


class WebsocketClientPolicy:
    """PI0 websocket inference client."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        proxy: str | bool | None = None,
        open_timeout: float = 10.0,
    ) -> None:
        self._msgpack = import_msgpack()
        self._websockets_sync_client = import_websockets_sync_client()
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = functools.partial(self._msgpack.Packer, default=pack_array)()
        self._unpackb = functools.partial(self._msgpack.unpackb, object_hook=unpack_array)
        self._api_key = api_key
        self._proxy = proxy
        self._open_timeout = open_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for PI0 server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = self._websockets_sync_client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=self._proxy,
                    open_timeout=self._open_timeout,
                )
                metadata = self._unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logging.info("Still waiting for PI0 server: %s", exc)
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpackb(response)

    def reset(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy UR7e + XHand PI0 policy client")

    # PI0 weights and policy inference live on the remote server.
    parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="PI0 websocket server IP or ws:// URI")
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Let websockets use HTTP(S)/SOCKS proxy environment variables. Default is direct connection.",
    )
    parser.add_argument("--server-open-timeout", type=float, default=10.0)

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory containing meta/info.json. Overrides --dataset-root/--dataset-name.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt sent to PI0 server.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=float, default=60.0, help="Run duration in seconds")
    parser.add_argument(
        "--policy-input-mode",
        choices=["auto", "vanilla", "obs_ae", "structured_single_ae"],
        default="auto",
        help=(
            "Observation format sent to the PI0 server. Use obs_ae for "
            "pi0_xhand_tactile_obs_ae_full_finetune and structured_single_ae for "
            "pi0_xhand_tactile_structured_single_ae."
        ),
    )
    parser.add_argument(
        "--structured-history-offsets",
        type=str,
        default=",".join(str(x) for x in STRUCTURED_TACTILE_HISTORY_OFFSETS),
        help="Comma-separated frame offsets for structured_single_ae history, matching training.",
    )

    parser.add_argument("--arm-ip", type=str, default="192.168.1.102")
    parser.add_argument("--arm-control-freq", type=float, default=15.0)
    parser.add_argument("--arm-max-relative-target", type=float, default=0.08)

    parser.add_argument("--hand-protocol", choices=["RS485", "EtherCAT"], default="EtherCAT")
    parser.add_argument("--hand-serial-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--hand-ethercat-interface", default="")
    parser.add_argument("--hand-control-freq", type=float, default=None)

    parser.add_argument("--realsense-front-serial", type=str, default="347622074420")
    parser.add_argument("--realsense-left-serial", type=str, default="347622075196")
    parser.add_argument("--realsense-right-serial", type=str, default="409122273624")
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=15)
    parser.add_argument("--camera-read-timeout-ms", type=float, default=80.0)

    parser.add_argument("--query-frequency", type=int, default=48, help="Request a new action chunk every N frames.")
    parser.add_argument("--max-action-chunk-size", type=int, default=30, help="Max actions to use from each server chunk.")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--no-home", action="store_true", help="Do not reset robot to home before policy run")
    parser.add_argument("--dry-run", action="store_true", help="Run inference but do not send actions")
    parser.add_argument("--check-config", action="store_true", help="Validate dataset wiring and exit")
    parser.add_argument("--yes", action="store_true", help="Skip interactive start confirmation")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    return (args.dataset_root / args.dataset_name).expanduser().resolve()


def load_feature_names(dataset_dir: Path) -> tuple[list[str], list[str], list[str]]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        logging.warning("Dataset info not found at %s; using built-in UR7e + XHand feature order", info_path)
        return FALLBACK_STATE_NAMES, FALLBACK_ACTION_NAMES, ["cam_front", "cam_left", "cam_right"]

    info = read_json(info_path)
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")
    camera_names = sorted(
        key.removeprefix("observation.images.")
        for key, ft in features.items()
        if key.startswith("observation.images.") and ft.get("dtype") in {"image", "video"}
    )

    if not state_names:
        logging.warning("observation.state names not found in %s; using built-in state order", info_path)
        state_names = FALLBACK_STATE_NAMES
    if not action_names:
        logging.warning("action names not found in %s; using built-in action order", info_path)
        action_names = FALLBACK_ACTION_NAMES
    if not camera_names:
        camera_names = ["cam_front", "cam_left", "cam_right"]

    return list(state_names), list(action_names), camera_names


def build_robot(args: argparse.Namespace):
    try:
        camera_configs = importlib.import_module("lerobot.cameras.configs")
        realsense = importlib.import_module("lerobot.cameras.realsense")
        ur7e_config = importlib.import_module("lerobot.robots.ur7e.ur7e_config")
        ur7e_xhand = importlib.import_module("lerobot.robots.ur7e_xhand.ur7e_xhand")
        ur7e_xhand_config = importlib.import_module("lerobot.robots.ur7e_xhand.ur7e_xhand_config")
        xhand_config = importlib.import_module("lerobot.robots.xhand.xhand_config")
    except ImportError as exc:
        raise ImportError(
            "Missing LeRobot UR7e/XHand deployment modules. Run this script in the robot deployment environment."
        ) from exc

    common_camera = dict(
        fps=args.realsense_fps,
        width=args.realsense_width,
        height=args.realsense_height,
        color_mode=camera_configs.ColorMode.RGB,
        use_depth=False,
    )
    cameras = {
        "cam_front": realsense.RealSenseCameraConfig(
            serial_number_or_name=args.realsense_front_serial,
            **common_camera,
        ),
        "cam_left": realsense.RealSenseCameraConfig(
            serial_number_or_name=args.realsense_left_serial,
            **common_camera,
        ),
        "cam_right": realsense.RealSenseCameraConfig(
            serial_number_or_name=args.realsense_right_serial,
            **common_camera,
        ),
    }

    robot_config = ur7e_xhand_config.UR7eXHandConfig(
        arm_config=ur7e_config.UR7eConfig(
            robot_ip=args.arm_ip,
            control_mode="servoj",
            speed=0.9,
            acceleration=0.5,
            servo_time=1.0 / args.arm_control_freq,
            servo_lookahead_time=1.0 / args.arm_control_freq,
            max_relative_target=args.arm_max_relative_target,
            cameras={},
        ),
        hand_config=xhand_config.XHandConfig(
            protocol=args.hand_protocol,
            serial_port=args.hand_serial_port,
            ethercat_interface=args.hand_ethercat_interface,
            control_frequency=args.hand_control_freq or args.fps,
            cameras={},
        ),
        cameras=cameras,
        synchronize_actions=True,
        action_timeout=max(0.2, 2.0 / args.arm_control_freq),
        camera_read_timeout_ms=args.camera_read_timeout_ms,
        check_arm_hand_collision=True,
        emergency_stop_both=True,
    )
    return ur7e_xhand.UR7eXHand(robot_config)


def build_env_state(obs: dict, state_names: list[str]) -> np.ndarray:
    missing = [name for name in state_names if name not in obs]
    if missing:
        raise KeyError(f"Missing observation state keys: {missing}")
    return np.array([obs[name] for name in state_names], dtype=np.float32)


def parse_history_offsets(offsets: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in offsets.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --structured-history-offsets={offsets!r}") from exc
    if not parsed:
        raise ValueError("--structured-history-offsets must contain at least one offset")
    if parsed[-1] != 0:
        raise ValueError("--structured-history-offsets must end with 0 for the current frame")
    if any(offset > 0 for offset in parsed):
        raise ValueError("--structured-history-offsets must only contain history/current offsets <= 0")
    return parsed


class StateHistoryBuffer:
    """Stores raw full-state frames and samples the training-time tactile history offsets."""

    def __init__(self, offsets: tuple[int, ...]):
        self._offsets = offsets
        self._max_age = abs(min(offsets))
        self._frames: collections.deque[tuple[int, np.ndarray]] = collections.deque(maxlen=self._max_age + 1)

    def append(self, frame_idx: int, state: np.ndarray) -> None:
        self._frames.append((frame_idx, np.asarray(state, dtype=np.float32).copy()))

    def sample(self, current_frame_idx: int) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("State history is empty; append the current state before sampling.")

        by_frame = {frame_idx: state for frame_idx, state in self._frames}
        earliest_frame, earliest_state = self._frames[0]
        latest_frame, latest_state = self._frames[-1]

        sampled = []
        for offset in self._offsets:
            target_frame = current_frame_idx + offset
            if target_frame <= earliest_frame:
                sampled.append(earliest_state)
            elif target_frame >= latest_frame:
                sampled.append(latest_state)
            else:
                sampled.append(by_frame.get(target_frame, latest_state))
        return np.stack(sampled, axis=0).astype(np.float32)


def extract_current_calc_force(state: np.ndarray, state_names: list[str]) -> np.ndarray | None:
    name_to_idx = {name: idx for idx, name in enumerate(state_names)}
    tactile = np.zeros((TACTILE_SENSOR_COUNT, len(TACTILE_AXES)), dtype=np.float32)
    found_all_named = True
    for sensor_id in range(TACTILE_SENSOR_COUNT):
        for axis_id, axis in enumerate(TACTILE_AXES):
            name = f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}"
            idx = name_to_idx.get(name)
            if idx is None:
                found_all_named = False
                break
            tactile[sensor_id, axis_id] = state[idx]
        if not found_all_named:
            break
    if found_all_named:
        return tactile

    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    if state.shape[-1] < required_dim:
        return None

    return np.stack(
        [
            state[
                TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_CALC_FORCE_OFFSET : TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_CALC_FORCE_OFFSET
                + 3
            ]
            for sensor_id in range(TACTILE_SENSOR_COUNT)
        ],
        axis=0,
    ).astype(np.float32)


def state_schema_has_calc_force(state_names: list[str]) -> bool:
    name_set = set(state_names)
    has_named_calc_force = all(
        f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}" in name_set
        for sensor_id in range(TACTILE_SENSOR_COUNT)
        for axis in TACTILE_AXES
    )
    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    return has_named_calc_force or len(state_names) >= required_dim


def infer_policy_input_mode(requested_mode: str, metadata: dict | None = None) -> str:
    if requested_mode != "auto":
        return requested_mode
    metadata_text = json.dumps(metadata or {}, default=str).lower()
    if "structured_single_ae" in metadata_text or "futuretactile" in metadata_text or "future_tactile" in metadata_text:
        return "structured_single_ae"
    if "tactile_obs" in metadata_text or "use_tactile_observation" in metadata_text:
        return "obs_ae"
    return "obs_ae"


def get_current_action(obs: dict, action_names: list[str]) -> np.ndarray:
    missing = [name for name in action_names if name not in obs]
    if missing:
        raise KeyError(f"Missing current action keys in observation: {missing}")
    return np.array([obs[name] for name in action_names], dtype=np.float32)


def get_image(obs: dict, camera_name: str, height: int, width: int) -> np.ndarray:
    image = obs.get(camera_name)
    if image is None:
        logging.warning("Camera %s missing from observation; sending a zero image", camera_name)
        return np.zeros((height, width, 3), dtype=np.uint8)
    return np.asarray(image, dtype=np.uint8)


def build_pi0_observation(
    obs: dict,
    env_state: np.ndarray,
    state_names: list[str],
    camera_names: list[str],
    args: argparse.Namespace,
    current_action_step: int,
    policy_input_mode: str,
    state_history: StateHistoryBuffer | None = None,
    frame_idx: int | None = None,
) -> dict:
    if policy_input_mode == "structured_single_ae":
        if state_history is None or frame_idx is None:
            raise ValueError("structured_single_ae mode requires state_history and frame_idx")
        policy_state = state_history.sample(frame_idx)
    else:
        policy_state = env_state

    observation = {
        "observation/state": policy_state,
        "prompt": args.prompt or args.task,
        "current_action_step": current_action_step,
    }

    if policy_input_mode == "obs_ae":
        tactile = extract_current_calc_force(env_state, state_names)
        if tactile is not None:
            observation["observation/tactile"] = tactile
        else:
            logging.warning(
                "Could not extract current calc_force tactile locally; relying on server transform to extract it from state."
            )

    for camera_name in camera_names:
        image = get_image(obs, camera_name, args.realsense_height, args.realsense_width)
        observation[f"observation/{camera_name}_image"] = image

    return observation


def normalize_action_chunk(actions, expected_dim: int, max_action_chunk_size: int) -> np.ndarray:
    action_chunk = np.asarray(actions, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]
    if action_chunk.ndim != 2:
        raise ValueError(f"Server returned actions with invalid shape {action_chunk.shape}; expected [T, {expected_dim}]")
    if action_chunk.shape[1] != expected_dim:
        raise ValueError(f"Server returned action dim {action_chunk.shape[1]}, expected {expected_dim}")
    return action_chunk[:max_action_chunk_size]


def action_array_to_dict(action: np.ndarray, action_names: list[str]) -> dict[str, float]:
    if len(action) != len(action_names):
        raise ValueError(f"Action length {len(action)} does not match action names {len(action_names)}")
    return {name: float(value) for name, value in zip(action_names, action, strict=True)}


def make_hold_action_chunk(obs: dict, action_names: list[str], n_action_steps: int) -> np.ndarray:
    current_action = get_current_action(obs, action_names)
    return np.repeat(current_action[None, :], repeats=n_action_steps, axis=0)


def seed_action_target_fallback(robot) -> None:
    fallback = {key: 0.0 for key in robot.action_target_position_features}

    try:
        arm_obs = robot.arm.get_observation()
        arm_target_position = [arm_obs[f"ee_pose.{i:02d}"] for i in (3, 7, 11)]
        arm_target_orientation = [arm_obs[f"ee_pose.{i:02d}"] for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)]
        fallback.update(
            {f"arm_target_position_{i}": float(value) for i, value in enumerate(arm_target_position)}
        )
        fallback.update(
            {f"arm_target_orientation_{i}": float(value) for i, value in enumerate(arm_target_orientation)}
        )
    except Exception as exc:
        logging.warning("Could not seed arm target fallback from current pose: %s", exc)

    robot._last_action_target_values = fallback


def require_ethercat_permissions(args: argparse.Namespace) -> None:
    if args.hand_protocol != "EtherCAT" or args.check_config:
        return
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        python_bin = Path(sys.executable).resolve()
        raise SystemExit(
            "XHand EtherCAT requires root privileges for raw socket access.\n"
            "Run the script with the lefranx interpreter directly, for example:\n"
            f"  sudo -E {python_bin} {Path(__file__).resolve()} --hand-protocol EtherCAT ...\n"
            "Tip: keep the same deployment arguments you just used."
        )


def main() -> int:
    args = parse_args()
    init_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1]")
    if args.query_frequency <= 0:
        raise ValueError("--query-frequency must be positive")
    if args.max_action_chunk_size <= 0:
        raise ValueError("--max-action-chunk-size must be positive")
    require_ethercat_permissions(args)

    dataset_dir = resolve_dataset_dir(args)
    state_names, action_names, camera_names = load_feature_names(dataset_dir)
    supported_cameras = {"cam_front", "cam_left", "cam_right"}
    unsupported_cameras = set(camera_names) - supported_cameras
    if unsupported_cameras:
        raise ValueError(
            f"Dataset expects unsupported cameras {sorted(unsupported_cameras)}; "
            f"available cameras are {sorted(supported_cameras)}"
        )
    history_offsets = parse_history_offsets(args.structured_history_offsets)
    preflight_mode = infer_policy_input_mode(args.policy_input_mode, None)
    if preflight_mode in {"obs_ae", "structured_single_ae"} and not state_schema_has_calc_force(state_names):
        raise ValueError(
            f"{preflight_mode} requires calc_force tactile values in observation.state, but the loaded state schema "
            f"only has {len(state_names)} fields and no hand_tactile_sensor_*.calc_force.* names. "
            "Check --dataset-dir/--dataset-root/--dataset-name."
        )

    print("=== UR7e + XHand PI0 Deployment Client ===")
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Dataset: {dataset_dir}")
    print(f"State dim: {len(state_names)}")
    print(f"Action dim: {len(action_names)}")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Prompt: {args.prompt or args.task}")
    print(f"FPS: {args.fps}, duration: {args.duration}s")
    print(f"Policy input mode: {args.policy_input_mode}")
    print(f"Dry run: {args.dry_run}")

    if args.check_config:
        print("Config check passed. Exiting before server/robot connection.")
        return 0

    client = WebsocketClientPolicy(
        args.server_ip,
        args.server_port,
        api_key=args.api_key,
        proxy=True if args.use_env_proxy else None,
        open_timeout=args.server_open_timeout,
    )
    metadata = client.get_server_metadata()
    if metadata:
        print(f"Server metadata: {metadata}")
    policy_input_mode = infer_policy_input_mode(args.policy_input_mode, metadata)
    state_history = StateHistoryBuffer(history_offsets) if policy_input_mode == "structured_single_ae" else None
    print(f"Resolved policy input mode: {policy_input_mode}")
    if state_history is not None:
        print(f"Structured tactile history offsets: {history_offsets}")

    robot = build_robot(args)
    dt = 1.0 / args.fps
    frame_idx = 0
    previous_action = None
    action_chunk = None
    chunk_idx = 0
    n_action_steps = 1
    query_count = 0
    current_action_step = 0

    try:
        print("Connecting robot and cameras...")
        robot.connect(calibrate=False)
        if not robot.is_connected:
            raise RuntimeError("Robot failed to connect")

        if not args.no_home:
            print("Resetting robot to home...")
            if not robot.reset_to_home():
                print("Warning: reset_to_home reported a failure")
            time.sleep(2.0)

        seed_action_target_fallback(robot)

        if not args.yes:
            input("Press ENTER to start PI0 deployment. Use Ctrl+C to stop: ")

        print("Starting PI0 control loop...")
        start_run = time.perf_counter()
        while time.perf_counter() - start_run < args.duration:
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            env_state = build_env_state(obs, state_names)
            if state_history is not None:
                state_history.append(frame_idx, env_state)

            should_query = (
                action_chunk is None
                or chunk_idx >= n_action_steps
                or query_count % args.query_frequency == 0
            )
            if should_query:
                observation = build_pi0_observation(
                    obs=obs,
                    env_state=env_state,
                    state_names=state_names,
                    camera_names=camera_names,
                    args=args,
                    current_action_step=current_action_step,
                    policy_input_mode=policy_input_mode,
                    state_history=state_history,
                    frame_idx=frame_idx,
                )
                try:
                    infer_start = time.perf_counter()
                    inference_result = client.infer(observation)
                    action_chunk = normalize_action_chunk(
                        inference_result["actions"],
                        expected_dim=len(action_names),
                        max_action_chunk_size=args.max_action_chunk_size,
                    )
                    n_action_steps = action_chunk.shape[0]
                    infer_ms = (time.perf_counter() - infer_start) * 1000
                    policy_ms = inference_result.get("policy_timing", {}).get("infer_ms")
                    server_ms = inference_result.get("server_timing", {}).get("infer_ms")
                    print(
                        f"Got action chunk {action_chunk.shape}: "
                        f"client_roundtrip={infer_ms:.1f}ms, policy={policy_ms}, server={server_ms}",
                        flush=True,
                    )
                    current_action_step += 1
                except Exception as exc:
                    print(f"Inference failed; holding current pose: {exc}", flush=True)
                    action_chunk = make_hold_action_chunk(obs, action_names, n_action_steps)
                chunk_idx = 0

            query_count += 1
            raw_action = action_chunk[chunk_idx].copy()
            chunk_idx += 1

            if previous_action is not None:
                action = args.smoothing_alpha * raw_action + (1.0 - args.smoothing_alpha) * previous_action
            else:
                action = raw_action
            action = action * args.action_scale
            previous_action = action.copy()

            action_dict = action_array_to_dict(action, action_names)
            if args.dry_run:
                if frame_idx % max(args.fps, 1) == 0:
                    print(
                        f"[dry-run] frame={frame_idx} chunk={chunk_idx}/{n_action_steps} "
                        f"action_range=[{action.min():.3f}, {action.max():.3f}]"
                    )
            else:
                robot.send_action(action_dict)

            elapsed = time.perf_counter() - loop_start
            if frame_idx % max(args.fps, 1) == 0:
                print(f"frame={frame_idx} chunk={chunk_idx}/{n_action_steps} loop={elapsed * 1000:.1f}ms")
            if elapsed < dt:
                time.sleep(dt - elapsed)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopping PI0 deployment...")
    except Exception as exc:
        print(f"Error in PI0 control loop: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            try:
                robot.stop()
            finally:
                robot.disconnect()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
