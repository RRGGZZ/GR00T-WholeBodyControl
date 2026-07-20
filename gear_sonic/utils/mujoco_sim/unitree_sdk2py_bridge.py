"""Bridge between Unitree SDK2 DDS topics and the MuJoCo simulation.

Subscribes to low-level motor commands (body + hands) and publishes
simulated sensor state (joint pos/vel, IMU, odometry) back over DDS,
so the WBC policy sees the sim as a real robot.
"""

import sys
import threading
import time
from typing import Dict, Tuple

import numpy as np
import scipy.spatial.transform
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, OdoState_

class UnitreeSdk2Bridge:
    """
    This class is responsible for bridging the Unitree SDK2 with the Groot environment.
    It is responsible for sending and receiving messages to and from the Unitree SDK2.
    Both the body and hand are supported.
    """

    def __init__(self, config):
        # Note that we do not give the mjdata and mjmodel to the UnitreeSdk2Bridge.
        # It is unsafe and would be unflexible if we use a hand-plugged robot model

        robot_type = config["ROBOT_TYPE"]
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__OdoState_ as OdoState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")

        self.num_body_motor = config["NUM_MOTORS"]
        self.num_hand_motor = config.get("NUM_HAND_MOTORS", 0)
        self.use_sensor = config["USE_SENSOR"]

        self.have_imu_ = False
        self.have_frame_sensor_ = False

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        # Only create odo_state for supported robot types
        if "g1" in robot_type or "h1-2" in robot_type:
            self.odo_state = OdoState_default()
            self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_state_puber.Init()
        else:
            self.odo_state = None
            self.odo_state_puber = None
        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        self.left_hand_state = HandState_default()
        self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
        self.left_hand_state_puber.Init()
        self.right_hand_state = HandState_default()
        self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
        self.right_hand_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.left_hand_cmd = HandCmd_default()
        self.left_hand_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)
        self.right_hand_cmd = HandCmd_default()
        self.right_hand_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 1)

        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        self.reset()

    def reset(self):
        with self.low_cmd_lock:
            self.low_cmd_received = False
            self.new_low_cmd = False
        with self.left_hand_cmd_lock:
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
        with self.right_hand_cmd_lock:
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False

    def LowCmdHandler(self, msg):
        with self.low_cmd_lock:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True

    def LeftHandCmdHandler(self, msg):
        with self.left_hand_cmd_lock:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True

    def RightHandCmdHandler(self, msg):
        with self.right_hand_cmd_lock:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True

    def cmd_received(self):
        with self.low_cmd_lock:
            low_cmd_received = self.low_cmd_received
        with self.left_hand_cmd_lock:
            left_hand_cmd_received = self.left_hand_cmd_received
        with self.right_hand_cmd_lock:
            right_hand_cmd_received = self.right_hand_cmd_received
        return low_cmd_received or left_hand_cmd_received or right_hand_cmd_received

    def PublishLowState(self, obs: Dict[str, any]):
        # publish body state
        if self.use_sensor:
            raise NotImplementedError("Sensor data is not implemented yet.")
        else:
            for i in range(self.num_body_motor):
                self.low_state.motor_state[i].q = obs["body_q"][i]
                self.low_state.motor_state[i].dq = obs["body_dq"][i]
                self.low_state.motor_state[i].ddq = obs["body_ddq"][i]
                self.low_state.motor_state[i].tau_est = obs["body_tau_est"][i]

        if self.use_sensor and self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        else:
            # Get data from ground truth
            self.odo_state.position[:] = obs["floating_base_pose"][:3]
            self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
            self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
            self.odo_state.angular_velocity[:] = obs["floating_base_vel"][3:6]
            # quaternion: w, x, y, z
            self.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
            # angular velocity
            self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
            # linear acceleration
            self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

            self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
            self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        # acceleration: x, y, z (only available when frame sensor is enabled)
        if self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        self.low_state.tick = int(obs["time"] * 1e3)
        self.low_state_puber.Write(self.low_state)

        self.odo_state.tick = int(obs["time"] * 1e3)
        self.odo_state_puber.Write(self.odo_state)

        self.torso_imu_puber.Write(self.torso_imu_state)

        # publish hand state
        for i in range(self.num_hand_motor):
            self.left_hand_state.motor_state[i].q = obs["left_hand_q"][i]
            self.left_hand_state.motor_state[i].dq = obs["left_hand_dq"][i]
        self.left_hand_state_puber.Write(self.left_hand_state)

        for i in range(self.num_hand_motor):
            self.right_hand_state.motor_state[i].q = obs["right_hand_q"][i]
            self.right_hand_state.motor_state[i].dq = obs["right_hand_dq"][i]
        self.right_hand_state_puber.Write(self.right_hand_state)

    def GetAction(self) -> Tuple[np.ndarray, bool, bool]:
        with self.low_cmd_lock:
            body_q = [self.low_cmd.motor_cmd[i].q for i in range(self.num_body_motor)]
        with self.left_hand_cmd_lock:
            left_hand_q = [self.left_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.right_hand_cmd_lock:
            right_hand_q = [self.right_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.low_cmd_lock and self.left_hand_cmd_lock and self.right_hand_cmd_lock:
            is_new_action = self.new_low_cmd and self.new_left_hand_cmd and self.new_right_hand_cmd
            if is_new_action:
                self.new_low_cmd = False
                self.new_left_hand_cmd = False
                self.new_right_hand_cmd = False

        return (
            np.concatenate([body_q[:-7], left_hand_q, body_q[-7:], right_hand_q]),
            self.cmd_received(),
            is_new_action,
        )

    def PublishWirelessController(self):
        import pygame

        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            # Pack 16 button states into a single integer via bit-shifting
            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        import pygame

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                    "LT": 2, "RT": 5, "DX": 6, "DY": 7,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
                    "XBOX": 8, "LSB": 9, "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                    "LT": 4, "RT": 5,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 9, "RB": 10, "SELECT": 4, "START": 6,
                    "XBOX": 5, "LSB": 7, "RSB": 8,
                    "DYU": 11, "DYD": 12, "DXL": 13, "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        import mujoco
        from loguru import logger
        from termcolor import colored

        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                logger.info(
                    f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}"
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class AdamG1DecoderBridge:
    """G1 DDS-compatible bridge for Adam tracking mode.

    The deployment process still publishes Unitree G1 ``rt/lowcmd`` messages.
    They are used only to close the virtual G1 control loop. Adam's tracking
    reference comes from the deployment process's ``g1_debug`` output, which
    exposes the current source-motion frame before the G1 control policy.
    """

    def __init__(self, config):
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandState_ as HandState_default,
            unitree_hg_msg_dds__IMUState_ as IMUState_default,
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_ as LowState_default,
            unitree_hg_msg_dds__OdoState_ as OdoState_default,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            HandState_,
            IMUState_,
            LowCmd_,
            LowState_,
        )

        self.num_body_motor = int(config.get("ADAM_G1_NUM_MOTORS", 29))
        self.num_hand_motor = 0
        self.use_sensor = False

        self.default_g1_q = np.asarray(
            config.get("ADAM_G1_DEFAULT_DOF_POS", np.zeros(self.num_body_motor)),
            dtype=np.float32,
        )
        if self.default_g1_q.shape != (self.num_body_motor,):
            raise ValueError(
                f"ADAM_G1_DEFAULT_DOF_POS must contain {self.num_body_motor} values"
        )
        self.virtual_g1_q = self.default_g1_q.copy()
        self.virtual_g1_dq = np.zeros(self.num_body_motor, dtype=np.float32)

        self.reference_source = config.get("ADAM_G1_REFERENCE_SOURCE", "zmq")
        if self.reference_source not in {"zmq", "lowcmd"}:
            raise ValueError(
                f"Unsupported ADAM_G1_REFERENCE_SOURCE={self.reference_source!r}; "
                "expected 'zmq' or 'lowcmd'."
            )
        self.reference_timeout = float(config.get("ADAM_G1_REFERENCE_TIMEOUT", 0.5))
        self.reference_root_pos = np.asarray(
            config.get("ADAM_G1_DEFAULT_ROOT_POS", [0.0, 0.0, 0.793]), dtype=np.float32
        )
        self.reference_root_quat_wxyz = np.asarray(
            config.get("ADAM_G1_DEFAULT_ROOT_QUAT_WXYZ", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.reference_g1_q = self.default_g1_q.copy()
        self.reference_last_update = None
        self.reference_message_error_reported = False
        self.reference_subscriber = None
        if self.reference_source == "zmq":
            from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

            self.reference_subscriber = ZMQStateSubscriber(
                host=config.get("ADAM_G1_REFERENCE_ZMQ_HOST", "localhost"),
                port=int(config.get("ADAM_G1_REFERENCE_ZMQ_PORT", 5557)),
                topic=config.get("ADAM_G1_REFERENCE_ZMQ_TOPIC", "g1_debug"),
            )

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd_lock = threading.Lock()
        self.low_cmd_received = False
        self.new_low_cmd = False
        self.low_cmd_last_update = None

        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.odo_state = OdoState_default()
        self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
        self.odo_state_puber.Init()

        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        self.left_hand_state = HandState_default()
        self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
        self.left_hand_state_puber.Init()
        self.right_hand_state = HandState_default()
        self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
        self.right_hand_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.joystick = None

    def reset(self):
        with self.low_cmd_lock:
            self.low_cmd_received = False
            self.new_low_cmd = False
            self.virtual_g1_q[:] = self.default_g1_q
            self.virtual_g1_dq[:] = 0.0
            self.low_cmd_last_update = None
        self.reference_g1_q[:] = self.default_g1_q
        self.reference_last_update = None

    def LowCmdHandler(self, msg):
        with self.low_cmd_lock:
            self.low_cmd = msg
            for i in range(self.num_body_motor):
                self.virtual_g1_q[i] = msg.motor_cmd[i].q
                self.virtual_g1_dq[i] = msg.motor_cmd[i].dq
            self.low_cmd_received = True
            self.new_low_cmd = True
            self.low_cmd_last_update = time.monotonic()

    def cmd_received(self):
        with self.low_cmd_lock:
            return self.low_cmd_received

    def _poll_zmq_reference(self):
        message = self.reference_subscriber.get_msg(clear=True)
        if message is None:
            return
        try:
            dof_pos = np.asarray(message["body_q_target"], dtype=np.float32)
            root_pos = np.asarray(message["base_trans_target"], dtype=np.float32)
            root_quat = np.asarray(message["base_quat_target"], dtype=np.float32)
            if dof_pos.shape != (self.num_body_motor,):
                raise ValueError(f"body_q_target has shape {dof_pos.shape}")
            if root_pos.shape != (3,):
                raise ValueError(f"base_trans_target has shape {root_pos.shape}")
            if root_quat.shape != (4,):
                raise ValueError(f"base_quat_target has shape {root_quat.shape}")
            if not (np.all(np.isfinite(dof_pos)) and np.all(np.isfinite(root_pos))):
                raise ValueError("reference contains non-finite position values")
            quat_norm = float(np.linalg.norm(root_quat))
            if not np.isfinite(quat_norm) or quat_norm < 1e-6:
                raise ValueError("base_quat_target is not a valid quaternion")
        except (KeyError, ValueError) as exc:
            if not self.reference_message_error_reported:
                print(f"Invalid Adam G1 reference message on g1_debug: {exc}")
                self.reference_message_error_reported = True
            return

        self.reference_g1_q = dof_pos
        self.reference_root_pos = root_pos
        self.reference_root_quat_wxyz = root_quat / quat_norm
        self.reference_last_update = time.monotonic()
        self.reference_message_error_reported = False

    def get_g1_reference(self) -> dict[str, np.ndarray] | None:
        if self.reference_source == "zmq":
            self._poll_zmq_reference()
            last_update = self.reference_last_update
            if last_update is None or time.monotonic() - last_update > self.reference_timeout:
                return None
            return {
                "dof_pos": self.reference_g1_q.copy(),
                "root_pos": self.reference_root_pos.copy(),
                "root_quat_wxyz": self.reference_root_quat_wxyz.copy(),
            }

        with self.low_cmd_lock:
            if (
                self.low_cmd_last_update is None
                or time.monotonic() - self.low_cmd_last_update > self.reference_timeout
            ):
                return None
            return {
                "dof_pos": self.virtual_g1_q.copy(),
                "root_pos": self.reference_root_pos.copy(),
                "root_quat_wxyz": self.reference_root_quat_wxyz.copy(),
            }

    def get_g1_reference_dof_pos(self) -> np.ndarray | None:
        reference = self.get_g1_reference()
        return None if reference is None else reference["dof_pos"]

    def close(self):
        if self.reference_subscriber is not None:
            self.reference_subscriber.close()
            self.reference_subscriber = None

    def PublishLowState(self, obs: Dict[str, any]):
        with self.low_cmd_lock:
            body_q = self.virtual_g1_q.copy()
            body_dq = self.virtual_g1_dq.copy()

        for i in range(self.num_body_motor):
            self.low_state.motor_state[i].q = float(body_q[i])
            self.low_state.motor_state[i].dq = float(body_dq[i])
            self.low_state.motor_state[i].ddq = 0.0
            self.low_state.motor_state[i].tau_est = 0.0

        self.odo_state.position[:] = obs["floating_base_pose"][:3]
        self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
        self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
        self.odo_state.angular_velocity[:] = obs["floating_base_vel"][3:6]

        self.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
        self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
        self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

        self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
        self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        self.low_state.tick = int(obs["time"] * 1e3)
        self.low_state_puber.Write(self.low_state)

        self.odo_state.tick = int(obs["time"] * 1e3)
        self.odo_state_puber.Write(self.odo_state)
        self.torso_imu_puber.Write(self.torso_imu_state)
        self.left_hand_state_puber.Write(self.left_hand_state)
        self.right_hand_state_puber.Write(self.right_hand_state)

    def PublishWirelessController(self):
        return

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        raise NotImplementedError("Joystick publishing is not implemented for Adam tracking mode")


class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(self):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point = np.array([0, 0, 1])
        self.length = 0
        self.enable = True

    def Advance(self, pose):
        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        δx = self.point - pos
        f = self.kp_pos * (δx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        # Convert quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat)
        rotvec = rot.as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        import glfw

        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")
