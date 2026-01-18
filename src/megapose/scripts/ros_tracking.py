#!/usr/bin/env python3
# If you get zero callbacks AND the websocket closes (often code=1009),
# your Image messages are likely too large for rosbridge_websocket max_message_size.
# Use a smaller resolution, /compressed, or increase max_message_size in the rosbridge launch.

import argparse
import base64
import json
import threading
import time
from pathlib import Path

from PIL import Image
import cv2, torch
import numpy as np
import roslibpy
from scipy.spatial.transform import Rotation as R

from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)


# ---------- MegaPose helpers ----------
def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "m" 
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                mesh_path = fn
        if mesh_path:
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def load_object_data(data_path: Path):
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(example_dir: Path):
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def numpy_to_observation(rgb: np.ndarray, cam: CameraData) -> ObservationTensor:
    # if rgb.shape[:2] != cam.resolution:
    #     rgb = cv2.resize(rgb, (cam.resolution[1], cam.resolution[0]))
    return ObservationTensor.from_numpy(rgb, depth=None, K=cam.K)


def unwrap_estimates(pred):
    return pred if hasattr(pred, "infos") else pred.get("iteration=1", pred)


# ---------- ROS image decode ----------
def _bytes_from_ros_data(data_field) -> bytes:
    if isinstance(data_field, str):
        return base64.b64decode(data_field)
    if isinstance(data_field, (bytes, bytearray)):
        return bytes(data_field)
    if isinstance(data_field, list):
        return bytes((int(x) & 0xFF) for x in data_field)
    raise TypeError(f"Unsupported image data field type: {type(data_field)}")

def load_observation(
    example_dir: Path,
    image_filename: str = "image_rgb.png" # 默认为单帧模式的文件名
):
    """
    加载指定路径的 RGB 图像和通用的相机参数
    """
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    # 修改：支持读取特定的图片文件
    img_path = example_dir / image_filename
    if not img_path.exists():
        # 尝试去 video 子文件夹找 (如果用户整理了数据集结构)
        img_path = example_dir / "video" / image_filename
    
    assert img_path.exists(), f"Image not found: {img_path}"

    rgb = np.array(Image.open(img_path), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution
    depth = None 
    return rgb, depth, camera_data
def load_observation_tensor(
    example_dir: Path,
    image_filename: str = "image_rgb.png"
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, image_filename)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation

def decode_ros_image_to_rgb_and_header(msg: dict):
    """Return (rgb_uint8[h,w,3], header_dict) or (None, None)."""
    if "data" not in msg:
        return None, None

    header = msg.get("header", {"frame_id": "camera_frame", "stamp": {}})

    raw = _bytes_from_ros_data(msg["data"])

    # sensor_msgs/CompressedImage
    if "format" in msg and "height" not in msg:
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), header

    # sensor_msgs/Image
    h = int(msg.get("height", 0))
    w = int(msg.get("width", 0))
    if h <= 0 or w <= 0:
        return None, None

    enc = (msg.get("encoding") or "").lower()
    step = int(msg.get("step") or 0)

    if enc in ("rgb8", "bgr8"):
        c = 3
    elif enc in ("rgba8", "bgra8"):
        c = 4
    elif "mono8" in enc or enc == "8uc1":
        c = 1
    else:
        c = 3  # fallback

    arr = np.frombuffer(raw, np.uint8)
    row_bytes = step if step else (w * c)
    need = h * row_bytes
    if arr.size < need:
        return None, None

    arr2d = arr[:need].reshape((h, row_bytes))
    pix = arr2d[:, : w * c].reshape((h, w, c))

    if enc == "bgr8":
        rgb = cv2.cvtColor(pix, cv2.COLOR_BGR2RGB)
    elif enc == "bgra8":
        rgb = cv2.cvtColor(pix, cv2.COLOR_BGRA2RGB)
    elif enc == "rgba8":
        rgb = cv2.cvtColor(pix, cv2.COLOR_RGBA2RGB)
    elif c == 1:
        rgb = cv2.cvtColor(pix, cv2.COLOR_GRAY2RGB)
    else:
        rgb = pix

    return rgb, header


def norm_stamp(stamp: dict, ros2: bool) -> dict:
    stamp = stamp or {}
    if ros2:
        return {
            "sec": int(stamp.get("sec", stamp.get("secs", 0))),
            "nanosec": int(stamp.get("nanosec", stamp.get("nsecs", 0))),
        }
    return {
        "secs": int(stamp.get("secs", stamp.get("sec", 0))),
        "nsecs": int(stamp.get("nsecs", stamp.get("nanosec", 0))),
    }


# ---------- Node ----------
class MegaPoseRosNode:
    def __init__(self, example_dir: Path, model_name: str, host: str, port: int, image_topic: str, ros2: bool, mesh_scale: float):
        self.example_dir = example_dir
        self.topic_name = image_topic
        self.ros2 = ros2
        self.mesh_scale = float(mesh_scale)

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_header = None
        self.new_image = threading.Event()
        self.msg_count = 0

        # MegaPose
        logger.info(f"Loading MegaPose model: {model_name}")
        self.camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
        self.object_dataset = make_object_dataset(example_dir)
        self.pose_estimator = load_named_model(model_name, self.object_dataset).cuda()
        self.model_info = NAMED_MODELS[model_name]
        self.detections = load_detections(example_dir).cuda()
        self.prev_pose = None
        self.initialized = False

        # ROS bridge
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.on("error", lambda *a: logger.error(f"ROSBridge error: {a}"))
        self.client.on("close", lambda *a: logger.warning(f"ROSBridge closed: {a}"))

        # Publishers (created on_ready)
        self.pose_pub = None
        self.robot_desc_pub = None
        self.tf_pub = None

        # Subscriptions: try raw + compressed, ROS1 + ROS2 type strings
        raw = self.topic_name
        comp = self.topic_name.rstrip("/") + "/compressed"
        self._subs = [
            (raw, "sensor_msgs/Image", "none"),
            (raw, "sensor_msgs/msg/Image", "none"),
            (raw, "sensor_msgs/Image", "png"),         # rosbridge PNG compression of raw Image
            (raw, "sensor_msgs/msg/Image", "png"),
            (comp, "sensor_msgs/CompressedImage", "none"),
            (comp, "sensor_msgs/msg/CompressedImage", "none"),
        ]
        self._sub_i = 0
        self.image_topic = None

        self.client.on_ready(self._on_ready, run_in_thread=True)

    def _msg_type(self, ros1: str, ros2: str) -> str:
        return ros2 if self.ros2 else ros1

    def _on_ready(self):
        logger.info("✅ Connected to ROSBridge.")

        self.pose_pub = roslibpy.Topic(
            self.client, "/megapose/data", self._msg_type("std_msgs/String", "std_msgs/msg/String")
        )
        self.pose_pub.advertise()

        self.robot_desc_pub = roslibpy.Topic(
            self.client, "/robot_description", self._msg_type("std_msgs/String", "std_msgs/msg/String"), latch=True
        )
        self.robot_desc_pub.advertise()
        self._publish_robot_description_once()

        self.tf_pub = roslibpy.Topic(
            self.client, "/tf", self._msg_type("tf2_msgs/TFMessage", "tf2_msgs/msg/TFMessage")
        )
        self.tf_pub.advertise()

        self.odom_pub = roslibpy.Topic(
            self.client, "/megapose/odom", self._msg_type("nav_msgs/Odometry", "nav_msgs/msg/Odometry")
        )
        self.odom_pub.advertise()

        self._subscribe_attempt(0)
        threading.Thread(target=self._worker, daemon=True).start()
        self._watchdog()  # start the retry loop

    def _subscribe_attempt(self, idx: int):
        if self.image_topic and getattr(self.image_topic, "is_subscribed", False):
            try:
                self.image_topic.unsubscribe()
            except Exception:
                pass

        topic, mtype, comp = self._subs[idx]
        self.image_topic = roslibpy.Topic(self.client, topic, mtype, compression=comp, queue_length=1)
        self.image_topic.subscribe(self._image_cb)
        logger.info(f"Subscribing: topic={topic} type={mtype} compression={comp}")

    def _watchdog(self):
        if not self.client.is_connected:
            return
        if self.msg_count == 0:
            self._sub_i = (self._sub_i + 1) % len(self._subs)
            logger.warning("No images yet -> trying next subscription variant...")
            self._subscribe_attempt(self._sub_i)
        t = threading.Timer(3.0, self._watchdog)
        t.daemon = True
        t.start()

    def _image_cb(self, msg):
        try:
            rgb, header = decode_ros_image_to_rgb_and_header(msg)
        except Exception as e:
            logger.error(f"Decode error: {e}")
            return
        if rgb is None:
            return

        self.msg_count += 1
        if self.msg_count == 1:
            logger.info("✅ First image received.")

        with self.lock:
            self.latest_rgb = rgb
            self.latest_header = header
        self.new_image.set()

    def _worker(self):
        while self.client.is_connected:
            if not self.new_image.wait(timeout=1.0):
                continue
            self.new_image.clear()

            with self.lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                header = dict(self.latest_header or {})
            if rgb is None:
                continue

            try:
                self._process_frame(rgb, header)
            except Exception as e:
                logger.error(f"Processing error: {e}")

    def _process_frame(self, rgb: np.ndarray, header: dict):
        with torch.inference_mode():
            start = time.perf_counter()
            # obs = numpy_to_observation(rgb, self.camera_data).cuda()

            img_name = self.image_files[self.idx].name
            obs = load_observation_tensor(example_dir, image_filename=img_name).cuda()
            self.idx += 1

            if not self.initialized:
                out, _ = self.pose_estimator.run_inference_pipeline(
                    obs, detections=self.detections, **self.model_info["inference_parameters"]
                )
                self.prev_pose = out
                self.initialized = True
                logger.info("✅ Tracker initialized.")
                self._publish_results(out, header)
                return
            
            # inp = unwrap_estimates(self.prev_pose)
            refiner_iterations = 1
            pred, _ = self.pose_estimator.forward_refiner(observation=obs, data_TCO_input=self.prev_pose, n_iterations=1)
            print(time.perf_counter()-start)
            self.prev_pose = pred[f"iteration={refiner_iterations}"]
            self._publish_results(pred, header)

    # -------- Robot model (URDF) --------
    def _iter_mesh_entries(self):
        meshes_dir = self.example_dir / "meshes"
        if not meshes_dir.exists():
            return []
        entries = []
        for p in meshes_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".obj", ".ply", ".stl", ".dae"):
                label = p.parent.name if p.parent != meshes_dir else p.stem
                entries.append((label, p))
        # keep one mesh per label (first one wins)
        seen, out = set(), []
        for label, p in entries:
            if label not in seen:
                seen.add(label)
                out.append((label, p))
        return out

    def _build_urdf(self, entries):
        s = self.mesh_scale
        root = "megapose_root"
        parts = [f'<?xml version="1.0"?>', f'<robot name="megapose_objects">', f'  <link name="{root}"/>\n']
        for label, mesh_path in entries:
            uri = f"file://{mesh_path.resolve()}"  # viewer must be able to access this path
            parts += [
                f'  <link name="{label}">',
                f'    <visual>',
                f'      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{uri}" scale="{s} {s} {s}"/></geometry>',
                f'    </visual>',
                f'  </link>',
                f'  <joint name="{root}_to_{label}" type="fixed">',
                f'    <parent link="{root}"/>',
                f'    <child link="{label}"/>',
                f'    <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'  </joint>\n',
            ]
        parts.append("</robot>\n")
        return "\n".join(parts)

    def _publish_robot_description_once(self):
        entries = self._iter_mesh_entries()
        if not entries:
            logger.warning("No mesh files found under example_dir/meshes; /robot_description not published.")
            return
        urdf = self._build_urdf(entries)
        self.robot_desc_pub.publish(roslibpy.Message({"data": urdf}))
        logger.info(f"✅ Published /robot_description with {len(entries)} mesh link(s).")

    # -------- TF + outputs --------
    def _publish_tf(self, parent_frame: str, child_frame: str, stamp: dict, t_xyz, q_xyzw):
        tf_msg = {
            "transforms": [
                {
                    "header": {"stamp": norm_stamp(stamp, self.ros2), "frame_id": parent_frame},
                    "child_frame_id": child_frame,
                    "transform": {
                        "translation": {"x": float(t_xyz[0]), "y": float(t_xyz[1]), "z": float(t_xyz[2])},
                        "rotation": {
                            "x": float(q_xyzw[0]),
                            "y": float(q_xyzw[1]),
                            "z": float(q_xyzw[2]),
                            "w": float(q_xyzw[3]),
                        },
                    },
                }
            ]
        }
        self.tf_pub.publish(roslibpy.Message(tf_msg))

    def _publish_results(self, pred, header: dict):
        est = unwrap_estimates(pred)
        labels = list(est.infos["label"])
        poses = est.poses.detach().cpu().numpy()

        parent_frame = header.get("frame_id", "camera_frame")
        stamp = (header.get("stamp") or {})

        out = []
        for label, T in zip(labels, poses):
            t = T[:3, 3]
            q = R.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]

            # TF: camera_frame -> <label>
            self._publish_tf(parent_frame, str(label), stamp, t, q)

            out.append(
                {
                    "label": str(label),
                    "position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
                    "orientation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
                }
            )

        self.pose_pub.publish(roslibpy.Message({"data": json.dumps(out)}))

        odom_msg = {
            "header": {"stamp": norm_stamp(stamp, self.ros2), "frame_id": parent_frame},  # pose is in camera frame
            "child_frame_id": str(label),
            "pose": {
                "pose": {
                    "position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
                    "orientation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
                },
                "covariance": [0.0] * 36,
            },
            "twist": {
                "twist": {
                    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "covariance": [0.0] * 36,
            },
        }
        self.odom_pub.publish(roslibpy.Message(odom_msg))


    def start(self):
        logger.info(f"Connecting + spinning ROSBridge... (topic: {self.topic_name})")
        self.client.run_forever()


if __name__ == "__main__":
    set_logging_level("info")
    ap = argparse.ArgumentParser()
    ap.add_argument("example_name")
    ap.add_argument("--model", default="megapose-1.0-RGB-multi-hypothesis")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--image_topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--ros2", action="store_true", help="Use ROS 2 message type strings for publishers (/tf, /robot_description, etc.)")
    ap.add_argument("--mesh_scale", type=float, default=1.0, help="URDF mesh scale (set 0.001 if your OBJ is in mm)")
    args = ap.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name
    node = MegaPoseRosNode(example_dir, args.model, args.host, args.port, args.image_topic, args.ros2, args.mesh_scale)

    try:
        node.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
