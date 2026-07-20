"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (
    AdamG1DecoderBridge,
    ElasticBand,
    UnitreeSdk2Bridge,
)
from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.mujoco_sim.adam_onnx_policy import (
    AdamG1Retargeter,
    AdamOnnxPolicy,
    AdamTrackingOnnxPolicy,
    AdamTrackingReferenceBuilder,
)

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ADAM_REFERENCE_SKELETON_EDGES = (
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 26),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 27),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 28),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 24),
    (15, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 25),
)


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.robot_type = self.config.get("ROBOT_TYPE", "g1_29dof")
        self.is_adam_onnx = (
            self.robot_type == "adam_pro" and self.config.get("ADAM_CONTROL_MODE") == "onnx"
        )
        self.adam_policy_type = self.config.get("ADAM_POLICY_TYPE", "locomotion")
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen
        self.elastic_band = None
        self.adam_policy = None
        self.adam_policy_counter = 0
        self.adam_policy_decimation = 1
        self.adam_target_q = None
        self.adam_tracking_body_ids = None
        self.adam_tracking_extend_parent_ids = None
        self.adam_tracking_extend_pos = None
        self.adam_tracking_extend_rot_xyzw = None
        self.adam_tracking_reference_active = False
        self.adam_reference_visualization_enabled = bool(
            self.is_adam_onnx
            and self.adam_policy_type == "tracking"
            and self.config.get("ADAM_REFERENCE_VISUALIZATION", True)
        )
        self.adam_reference_marker_size = float(
            self.config.get("ADAM_REFERENCE_MARKER_SIZE", 0.025)
        )
        self.adam_reference_link_radius = float(
            self.config.get("ADAM_REFERENCE_LINK_RADIUS", 0.008)
        )
        self.adam_reference_color = np.asarray(
            self.config.get("ADAM_REFERENCE_COLOR", [0.1, 1.0, 0.25, 0.85]),
            dtype=np.float32,
        )
        if self.adam_reference_color.shape != (4,):
            raise ValueError("ADAM_REFERENCE_COLOR must contain four RGBA values")

        self.init_scene()
        if self.is_adam_onnx:
            self.init_adam_controller()
            self.torques = np.zeros(self.mj_model.nu)
            self.torque_limit = self.adam_torque_limit.copy()
            self.reset()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_body = self.config.get("TORSO_BODY", "torso_link")
        self.torso_index = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.torso_body
        )
        if self.torso_index == -1:
            raise ValueError(f"Torso body '{self.torso_body}' not found in MuJoCo model")
        self.root_body = self.config.get("ROOT_BODY", "pelvis")
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        root_joint_candidates = [
            self.config.get("FLOATING_BASE_JOINT", "floating_base_joint"),
            "floating_base_joint",
            "floating_base",
        ]
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        # Check for static root link (fixed base)
        self.use_floating_root_link = any(name in joint_names for name in root_joint_candidates)
        self.use_constrained_root_link = "constrained_base_joint" in joint_names

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if self.config["ROBOT_TYPE"] == "adam_pro":
                self.band_attached_link = self.mj_model.body(self.root_body).id
            elif "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self._mujoco_key_callback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self._mujoco_key_callback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body(self.root_body).id

        if self.is_adam_onnx:
            self.body_joint_index = np.array(
                [
                    self.mj_model.joint(name).id
                    for name in self.config["ADAM_ACTUATOR_NAMES"]
                    if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) != -1
                ]
            )
            self.left_hand_index = np.array([], dtype=int)
            self.right_hand_index = np.array([], dtype=int)
        else:
            self.body_joint_index = []
            self.left_hand_index = []
            self.right_hand_index = []
            for i in range(self.mj_model.njnt):
                name = self.mj_model.joint(i).name
                if any(
                    [
                        part_name in name
                        for part_name in [
                            "hip",
                            "knee",
                            "ankle",
                            "waist",
                            "shoulder",
                            "elbow",
                            "wrist",
                        ]
                    ]
                ):
                    self.body_joint_index.append(i)
                elif "left_hand" in name:
                    self.left_hand_index.append(i)
                elif "right_hand" in name:
                    self.right_hand_index.append(i)

            assert len(self.body_joint_index) == self.robot.NUM_JOINTS
            assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
            assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

            self.body_joint_index = np.array(self.body_joint_index)
            self.left_hand_index = np.array(self.left_hand_index)
            self.right_hand_index = np.array(self.right_hand_index)

    def _joint_qpos_qvel_indices(self, joint_name: str) -> tuple[int, int]:
        joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model")
        return int(self.mj_model.jnt_qposadr[joint_id]), int(self.mj_model.jnt_dofadr[joint_id])

    def init_adam_controller(self):
        self.adam_actuator_names = list(self.config["ADAM_ACTUATOR_NAMES"])
        if len(self.adam_actuator_names) != self.mj_model.nu:
            raise ValueError(
                f"Adam config has {len(self.adam_actuator_names)} actuators, "
                f"but MuJoCo model has {self.mj_model.nu}"
            )

        self.adam_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.adam_actuator_names
            ],
            dtype=int,
        )
        if np.any(self.adam_actuator_ids < 0):
            missing = [
                name for name, idx in zip(self.adam_actuator_names, self.adam_actuator_ids) if idx < 0
            ]
            raise ValueError(f"Adam actuator(s) not found in MuJoCo model: {missing}")

        self.adam_actuator_joint_names = []
        for actuator_id in self.adam_actuator_ids:
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            self.adam_actuator_joint_names.append(self.mj_model.joint(joint_id).name)

        self.adam_actuator_qpos_ids = np.array(
            [self._joint_qpos_qvel_indices(name)[0] for name in self.adam_actuator_joint_names],
            dtype=int,
        )
        self.adam_actuator_qvel_ids = np.array(
            [self._joint_qpos_qvel_indices(name)[1] for name in self.adam_actuator_joint_names],
            dtype=int,
        )
        self.adam_policy_joint_names = list(self.config["ADAM_POLICY_JOINT_NAMES"])
        self.adam_policy_qpos_ids = np.array(
            [self._joint_qpos_qvel_indices(name)[0] for name in self.adam_policy_joint_names],
            dtype=int,
        )
        self.adam_policy_qvel_ids = np.array(
            [self._joint_qpos_qvel_indices(name)[1] for name in self.adam_policy_joint_names],
            dtype=int,
        )

        self.adam_policy_to_actuator_ids = np.array(
            [self.adam_actuator_names.index(name) for name in self.adam_policy_joint_names],
            dtype=int,
        )
        self.adam_default_actuator_q = np.asarray(
            self.config["DEFAULT_MOTOR_ANGLES"], dtype=np.float32
        )
        if self.adam_policy_type == "tracking" and "ADAM_TRACKING_DEFAULT_MOTOR_ANGLES" in self.config:
            self.adam_default_actuator_q = np.asarray(
                self.config["ADAM_TRACKING_DEFAULT_MOTOR_ANGLES"], dtype=np.float32
            )
        self.adam_motor_kp = np.asarray(self.config["MOTOR_KP"], dtype=np.float32)
        self.adam_motor_kd = np.asarray(self.config["MOTOR_KD"], dtype=np.float32)
        self.adam_torque_limit = np.asarray(
            self.config["motor_effort_limit_list"], dtype=np.float32
        )
        self.adam_motor_pos_lower = np.asarray(
            self.config["motor_pos_lower_limit_list"], dtype=np.float32
        )
        self.adam_motor_pos_upper = np.asarray(
            self.config["motor_pos_upper_limit_list"], dtype=np.float32
        )

        if self.adam_policy_type == "locomotion":
            self.adam_policy = AdamOnnxPolicy(
                onnx_path=self.config["ADAM_POLICY_ONNX_PATH"],
                joint_names=self.adam_policy_joint_names,
                default_dof_pos=self.config["ADAM_POLICY_DEFAULT_DOF_POS"],
                kp=self.config["ADAM_POLICY_KP"],
                effort_limit=self.config["ADAM_POLICY_EFFORT_LIMIT"],
                control_dt=self.config.get("ADAM_POLICY_DT", 0.02),
                cycle_time=self.config.get("ADAM_POLICY_CYCLE_TIME", 0.8),
                command=self.config.get("ADAM_COMMAND", [0.0, 0.0, 0.0]),
            )
        elif self.adam_policy_type == "tracking":
            self._init_adam_tracking_controller()
        else:
            raise ValueError(
                f"Unsupported ADAM_POLICY_TYPE={self.adam_policy_type!r}; "
                "expected 'locomotion' or 'tracking'."
            )
        self.adam_policy_decimation = max(
            1, int(round(float(self.config.get("ADAM_POLICY_DT", 0.02)) / self.sim_dt))
        )
        self.adam_policy_counter = 0
        self.adam_target_q = self.adam_default_actuator_q.copy()

    def _init_adam_tracking_controller(self):
        tracking_body_names = list(self.config["ADAM_TRACKING_BODY_NAMES"])
        tracking_extend_config = list(self.config["ADAM_TRACKING_EXTEND_CONFIG"])

        self.adam_tracking_body_ids = np.array(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in tracking_body_names
            ],
            dtype=int,
        )
        if np.any(self.adam_tracking_body_ids < 0):
            missing = [
                name
                for name, idx in zip(tracking_body_names, self.adam_tracking_body_ids)
                if idx < 0
            ]
            raise ValueError(f"Adam tracking body(s) not found in MuJoCo model: {missing}")

        extend_parent_ids = []
        extend_pos = []
        extend_rot_xyzw = []
        for item in tracking_extend_config:
            parent_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, item["parent_name"]
            )
            if parent_id == -1:
                raise ValueError(
                    f"Adam tracking extend parent not found: {item['parent_name']}"
                )
            extend_parent_ids.append(parent_id)
            extend_pos.append(item["pos"])
            rot_wxyz = np.asarray(item["rot"], dtype=np.float32)
            extend_rot_xyzw.append(rot_wxyz[[1, 2, 3, 0]])
        self.adam_tracking_extend_parent_ids = np.asarray(extend_parent_ids, dtype=int)
        self.adam_tracking_extend_pos = np.asarray(extend_pos, dtype=np.float32)
        self.adam_tracking_extend_rot_xyzw = np.asarray(extend_rot_xyzw, dtype=np.float32)

        retargeter = AdamG1Retargeter(
            package_root=self.config.get(
                "ADAM_RETARGETING_ROOT", "/home/r/Downloads/robot_to_robot_retargeting"
            ),
            g1_default_dof_pos=self.config["ADAM_G1_DEFAULT_DOF_POS"],
            g1_default_root_pos=self.config.get("ADAM_G1_DEFAULT_ROOT_POS", [0.0, 0.0, 0.793]),
            g1_default_root_quat_wxyz=self.config.get(
                "ADAM_G1_DEFAULT_ROOT_QUAT_WXYZ", [1.0, 0.0, 0.0, 0.0]
            ),
            max_iter=self.config.get("ADAM_RETARGET_MAX_ITER", 5),
            solver=self.config.get("ADAM_RETARGET_SOLVER", "daqp"),
            damping=self.config.get("ADAM_RETARGET_DAMPING", 5e-1),
            retarget_every_n=self.config.get("ADAM_RETARGET_EVERY_N", 1),
            input_epsilon=self.config.get("ADAM_RETARGET_INPUT_EPSILON", 0.0),
            verbose=self.config.get("ADAM_RETARGET_VERBOSE", False),
        )
        reference_builder = AdamTrackingReferenceBuilder(
            adam_policy_joint_names=self.adam_policy_joint_names,
            fk_xml_path=self.config.get(
                "ADAM_TRACKING_FK_XML_PATH",
                "/home/r/Downloads/HumanoidVLA_MJ/example/python/humanoidverse/data/robots/adam_sp/adam_lite_Optim.xml",
            ),
            source_xml_path=str(
                pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"]
            ),
            body_names=tracking_body_names,
            extend_config=tracking_extend_config,
            fk_body_names=self.config.get("ADAM_TRACKING_FK_BODY_NAMES", tracking_body_names),
            fk_extend_config=self.config.get(
                "ADAM_TRACKING_FK_EXTEND_CONFIG", tracking_extend_config
            ),
            future_steps=self.config.get("ADAM_TRACKING_FUTURE_STEPS", 10),
        )
        self.adam_policy = AdamTrackingOnnxPolicy(
            onnx_path=self.config.get("ADAM_TRACKING_ONNX_PATH", self.config["ADAM_POLICY_ONNX_PATH"]),
            joint_names=self.adam_policy_joint_names,
            default_dof_pos=self.config.get(
                "ADAM_TRACKING_POLICY_DEFAULT_DOF_POS",
                self.config["ADAM_POLICY_DEFAULT_DOF_POS"],
            ),
            kp=self.config.get("ADAM_TRACKING_POLICY_KP", self.config["ADAM_POLICY_KP"]),
            effort_limit=self.config.get(
                "ADAM_TRACKING_POLICY_EFFORT_LIMIT",
                self.config["ADAM_POLICY_EFFORT_LIMIT"],
            ),
            retargeter=retargeter,
            reference_builder=reference_builder,
            control_dt=self.config.get("ADAM_POLICY_DT", 0.02),
            command=self.config.get("ADAM_COMMAND", [0.0, 0.0, 0.0]),
            history_len=self.config.get("ADAM_TRACKING_HISTORY_LEN", 10),
            future_steps=self.config.get("ADAM_TRACKING_FUTURE_STEPS", 10),
            clip_observations=self.config.get("ADAM_CLIP_OBSERVATIONS", 100.0),
            clip_actions=self.config.get("ADAM_CLIP_ACTIONS", 100.0),
        )

    def _get_adam_tracking_body_positions(self) -> np.ndarray:
        body_pos = self.mj_data.xpos[self.adam_tracking_body_ids].copy()
        if len(self.adam_tracking_extend_parent_ids) == 0:
            return body_pos.astype(np.float32)

        parent_pos = self.mj_data.xpos[self.adam_tracking_extend_parent_ids].copy()
        parent_rot_xyzw = self.mj_data.xquat[self.adam_tracking_extend_parent_ids][
            :, [1, 2, 3, 0]
        ].copy()
        rotated = np.array(
            [
                Rotation.from_quat(parent_rot_xyzw[i]).apply(self.adam_tracking_extend_pos[i])
                for i in range(len(self.adam_tracking_extend_parent_ids))
            ],
            dtype=np.float32,
        )
        extend_pos = np.array(
            [
                Rotation.from_quat(self.adam_tracking_extend_rot_xyzw[i]).apply(rotated[i])
                + parent_pos[i]
                for i in range(len(self.adam_tracking_extend_parent_ids))
            ],
            dtype=np.float32,
        )
        return np.concatenate([body_pos, extend_pos], axis=0).astype(np.float32)

    def _get_g1_decoder_reference(self) -> dict[str, np.ndarray] | None:
        if self.unitree_bridge is None:
            return None
        if hasattr(self.unitree_bridge, "get_g1_reference"):
            return self.unitree_bridge.get_g1_reference()
        if hasattr(self.unitree_bridge, "get_g1_reference_dof_pos"):
            dof_pos = self.unitree_bridge.get_g1_reference_dof_pos()
            if dof_pos is not None:
                return {"dof_pos": dof_pos, "root_pos": None, "root_quat_wxyz": None}
        return None

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def compute_adam_torques(self) -> np.ndarray:
        if self.adam_policy is None:
            raise RuntimeError("Adam ONNX policy is not initialized")

        if self.adam_policy_counter % self.adam_policy_decimation == 0:
            base_quat_xyzw = self.mj_data.qpos[3:7][[1, 2, 3, 0]].copy()
            base_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.mj_model,
                self.mj_data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.root_body_id,
                base_velocity,
                1,
            )
            base_ang_vel = base_velocity[:3].copy()
            policy_q = self.mj_data.qpos[self.adam_policy_qpos_ids].copy()
            policy_dq = self.mj_data.qvel[self.adam_policy_qvel_ids].copy()
            if self.adam_policy_type == "tracking":
                reference = self._get_g1_decoder_reference()
                if reference is None:
                    if self.adam_tracking_reference_active:
                        print("Adam PND reference timed out; holding the current pose")
                        self.adam_policy.reset()
                        self.adam_target_q = self.mj_data.qpos[
                            self.adam_actuator_qpos_ids
                        ].copy()
                    self.adam_tracking_reference_active = False
                    target_q = None
                else:
                    if not self.adam_tracking_reference_active:
                        self.adam_policy.reset()
                        print("Adam PND reference connected; tracking source motion")
                    self.adam_tracking_reference_active = True
                    target_q, _ = self.adam_policy.compute_target(
                        root_pos=self.mj_data.qpos[:3].copy(),
                        base_quat_xyzw=base_quat_xyzw,
                        base_ang_vel=base_ang_vel,
                        dof_pos=policy_q,
                        dof_vel=policy_dq,
                        current_body_positions=self._get_adam_tracking_body_positions(),
                        g1_reference_dof_pos=reference["dof_pos"],
                        g1_reference_root_pos=reference["root_pos"],
                        g1_reference_root_quat_wxyz=reference["root_quat_wxyz"],
                    )
            else:
                target_q, _ = self.adam_policy.compute_target(
                    base_quat_xyzw=base_quat_xyzw,
                    base_ang_vel=base_ang_vel,
                    dof_pos=policy_q,
                    dof_vel=policy_dq,
                )
            if target_q is not None:
                self.adam_target_q = self.adam_default_actuator_q.copy()
                self.adam_target_q[self.adam_policy_to_actuator_ids] = target_q
                self.adam_target_q = np.clip(
                    self.adam_target_q, self.adam_motor_pos_lower, self.adam_motor_pos_upper
                )
        self.adam_policy_counter += 1

        q = self.mj_data.qpos[self.adam_actuator_qpos_ids]
        dq = self.mj_data.qvel[self.adam_actuator_qvel_ids]
        torques = self.adam_motor_kp * (self.adam_target_q - q) - self.adam_motor_kd * dq
        return np.clip(torques, -self.adam_torque_limit, self.adam_torque_limit)

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body(self.torso_body).xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body(self.torso_body).xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body(self.torso_body).id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        if self.is_adam_onnx:
            obs["body_q"] = self.mj_data.qpos[self.adam_actuator_qpos_ids]
            obs["body_dq"] = self.mj_data.qvel[self.adam_actuator_qvel_ids]
            obs["body_ddq"] = self.mj_data.qacc[self.adam_actuator_qvel_ids]
            obs["body_tau_est"] = self.mj_data.actuator_force[self.adam_actuator_ids]
        else:
            obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
            obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
            obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
            obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        if self.unitree_bridge is not None:
            self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge is not None and self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        if self.is_adam_onnx:
            self.torques = self.compute_adam_torques()
        else:
            body_torques = self.compute_body_torques()
            hand_torques = self.compute_hand_torques()
            # -1: actuator array is 0-based while joint indices from the model are 1-based
            self.torques[self.body_joint_index - 1] = body_torques
            if self.num_hand_dof > 0:
                self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
                self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            with self.viewer.lock():
                self._update_adam_reference_visualization()
            self.viewer.sync()

    def _update_adam_reference_visualization(self):
        if not self.is_adam_onnx or self.adam_policy_type != "tracking":
            return
        scene = self.viewer.user_scn
        scene.ngeom = 0
        if not self.adam_reference_visualization_enabled or self.adam_policy is None:
            return

        positions = self.adam_policy.get_reference_body_positions()
        if positions is None:
            return
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Adam reference markers must have shape (N, 3), got {positions.shape}")

        identity = np.eye(3, dtype=np.float64).reshape(-1)
        marker_size = np.array(
            [self.adam_reference_marker_size, 0.0, 0.0], dtype=np.float64
        )
        for position in positions:
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=marker_size,
                pos=position,
                mat=identity,
                rgba=self.adam_reference_color,
            )
            scene.ngeom += 1

        for start_index, end_index in ADAM_REFERENCE_SKELETON_EDGES:
            if scene.ngeom >= scene.maxgeom:
                return
            if start_index >= len(positions) or end_index >= len(positions):
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.zeros(3, dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=identity,
                rgba=self.adam_reference_color,
            )
            mujoco.mjv_connector(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=self.adam_reference_link_radius,
                from_=positions[start_index],
                to=positions[end_index],
            )
            scene.ngeom += 1

    def _mujoco_key_callback(self, key):
        if self.elastic_band is not None:
            self.elastic_band.MujuocoKeyCallback(key)
        if key == ord("P"):
            self.handle_keyboard_button("p")

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key == "p" and self.is_adam_onnx and self.adam_policy_type == "tracking":
            self.adam_reference_visualization_enabled = (
                not self.adam_reference_visualization_enabled
            )
            state = "on" if self.adam_reference_visualization_enabled else "off"
            print(f"Adam PND reference visualization: {state}")
            return
        if self.is_adam_onnx and self.adam_policy is not None:
            command = np.asarray(self.config.get("ADAM_COMMAND", [0.0, 0.0, 0.0]), dtype=float)
            if key == "up":
                command[0] = min(command[0] + 0.1, 0.5)
            elif key == "down":
                command[0] = max(command[0] - 0.1, -0.5)
            elif key == "left":
                command[2] = min(command[2] + 0.1, 0.5)
            elif key == "right":
                command[2] = max(command[2] - 0.1, -0.5)
            elif key == "space":
                command[:] = 0.0
            self.config["ADAM_COMMAND"] = command.tolist()
            self.adam_policy.set_command(command)
            return
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.is_adam_onnx:
            root_pos = np.asarray(self.config.get("ADAM_DEFAULT_ROOT_POS", [0.0, 0.0, 0.9]))
            root_quat = np.asarray(
                self.config.get("ADAM_DEFAULT_ROOT_QUAT_WXYZ", [1.0, 0.0, 0.0, 0.0])
            )
            self.mj_data.qpos[:3] = root_pos
            self.mj_data.qpos[3:7] = root_quat
            self.mj_data.qpos[self.adam_actuator_qpos_ids] = self.adam_default_actuator_q
            self.mj_data.qvel[:] = 0.0
            self.mj_data.ctrl[:] = 0.0
            self.adam_target_q = self.adam_default_actuator_q.copy()
            self.adam_policy_counter = 0
            self.adam_tracking_reference_active = False
            if self.adam_policy is not None:
                self.adam_policy.reset()
            mujoco.mj_forward(self.mj_model, self.mj_data)


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        self.is_adam_onnx = (
            self.config.get("ROBOT_TYPE") == "adam_pro"
            and self.config.get("ADAM_CONTROL_MODE") == "onnx"
        )
        self.adam_policy_type = self.config.get("ADAM_POLICY_TYPE", "locomotion")

        if not (self.is_adam_onnx and self.adam_policy_type == "locomotion"):
            try:
                if self.config.get("INTERFACE", None):
                    ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
                else:
                    ChannelFactoryInitialize(self.config["DOMAIN_ID"])
            except Exception as e:
                print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        if self.is_adam_onnx and self.adam_policy_type == "locomotion":
            self.unitree_bridge = None
            return
        if self.is_adam_onnx and self.adam_policy_type == "tracking":
            self.unitree_bridge = AdamG1DecoderBridge(self.config)
            return
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.unitree_bridge is not None and hasattr(self.unitree_bridge, "close"):
                self.unitree_bridge.close()
        except Exception as e:
            print(f"Warning while closing simulator bridge: {e}")
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
