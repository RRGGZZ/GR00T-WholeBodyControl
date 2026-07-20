"""Adam Pro ONNX policy wrappers for locomotion and G1-reference tracking."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Sequence

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _as_float_array(value: Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _quat_heading_inv_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    rot_dir = Rotation.from_quat(quat_xyzw).apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    heading = np.arctan2(rot_dir[1], rot_dir[0])
    return Rotation.from_euler("z", -heading).as_quat().astype(np.float32)


def _quat_rotate_xyzw(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).apply(vec).astype(np.float32)


def _quat_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)


class AdamOnnxPolicy:
    """Runs an Adam locomotion ONNX policy and returns 23 target joint positions."""

    OBS_ORDER = [
        "actions",
        "base_ang_vel",
        "commands",
        "dof_pos",
        "dof_vel",
        "gait_phase",
        "history_actor",
        "projected_gravity",
    ]
    HISTORY_ORDER = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
    HISTORY_LENGTH = 4

    def __init__(
        self,
        onnx_path: str,
        joint_names: Sequence[str],
        default_dof_pos: Sequence[float],
        kp: Sequence[float],
        effort_limit: Sequence[float],
        control_dt: float = 0.02,
        cycle_time: float = 0.8,
        command: Sequence[float] | None = None,
        clip_observations: float = 100.0,
        clip_actions: float = 100.0,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Adam Pro mode requires onnxruntime. Install it in .venv_sim with "
                "`pip install onnxruntime`."
            ) from exc

        self.onnx_path = Path(onnx_path).expanduser()
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Adam ONNX policy not found: {self.onnx_path}")

        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        input_shapes = {
            inp.name: [dim if isinstance(dim, int) else -1 for dim in inp.shape]
            for inp in self.session.get_inputs()
        }
        if self.input_names != ["actor_obs"] or input_shapes["actor_obs"][-1] != 380:
            raise ValueError(
                "Adam locomotion policy expects single input `actor_obs [1, 380]`; "
                f"got inputs {input_shapes} from {self.onnx_path}."
            )

        self.joint_names = list(joint_names)
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float32)
        self.kp = np.asarray(kp, dtype=np.float32)
        self.effort_limit = np.asarray(effort_limit, dtype=np.float32)
        self.num_dof = len(self.joint_names)
        if self.default_dof_pos.shape != (self.num_dof,):
            raise ValueError("default_dof_pos must match Adam policy joint count")
        if self.kp.shape != (self.num_dof,):
            raise ValueError("kp must match Adam policy joint count")
        if self.effort_limit.shape != (self.num_dof,):
            raise ValueError("effort_limit must match Adam policy joint count")

        self.action_scale = 0.25 * self.effort_limit / self.kp
        self.control_dt = float(control_dt)
        self.cycle_time = float(cycle_time)
        self.command = (
            np.zeros(3, dtype=np.float32)
            if command is None
            else _as_float_array(command, (3,), "Adam command")
        )
        self.clip_observations = float(clip_observations)
        self.clip_actions = float(clip_actions)
        self.reset()

    def reset(self):
        self.timer = 0
        self.last_action = np.zeros(self.num_dof, dtype=np.float32)
        self.history = {
            "actions": np.zeros((self.HISTORY_LENGTH, self.num_dof), dtype=np.float32),
            "base_ang_vel": np.zeros((self.HISTORY_LENGTH, 3), dtype=np.float32),
            "dof_pos": np.zeros((self.HISTORY_LENGTH, self.num_dof), dtype=np.float32),
            "dof_vel": np.zeros((self.HISTORY_LENGTH, self.num_dof), dtype=np.float32),
            "projected_gravity": np.zeros((self.HISTORY_LENGTH, 3), dtype=np.float32),
        }

    def set_command(self, command: Sequence[float]):
        self.command = _as_float_array(command, (3,), "Adam command")

    def _gait_phase(self) -> np.ndarray:
        phase = self.timer * self.control_dt / self.cycle_time
        return np.array(
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
            dtype=np.float32,
        )

    def _scaled_current(
        self,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "actions": self.last_action.astype(np.float32),
            "base_ang_vel": (base_ang_vel * 0.25).astype(np.float32),
            "commands": self.command.astype(np.float32),
            "dof_pos": (dof_pos - self.default_dof_pos).astype(np.float32),
            "dof_vel": (dof_vel * 0.05).astype(np.float32),
            "gait_phase": self._gait_phase(),
            "projected_gravity": projected_gravity.astype(np.float32),
        }

    def _history_actor(self) -> np.ndarray:
        return np.concatenate(
            [self.history[key].reshape(-1) for key in self.HISTORY_ORDER]
        ).astype(np.float32)

    def _build_actor_obs(
        self,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pieces = self._scaled_current(base_ang_vel, projected_gravity, dof_pos, dof_vel)
        pieces["history_actor"] = self._history_actor()
        actor_obs = np.concatenate([pieces[key].reshape(-1) for key in self.OBS_ORDER])
        actor_obs = np.clip(actor_obs, -self.clip_observations, self.clip_observations)
        return actor_obs.reshape(1, -1).astype(np.float32), pieces

    def _update_history(self, pieces: dict[str, np.ndarray]):
        for key in self.HISTORY_ORDER:
            self.history[key][1:] = self.history[key][:-1]
            self.history[key][0] = pieces[key]

    def compute_target(
        self,
        base_quat_xyzw: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = Rotation.from_quat(base_quat_xyzw)
        projected_gravity = rotation.apply(np.array([0.0, 0.0, -1.0]), inverse=True)

        actor_obs, pieces = self._build_actor_obs(
            base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
            projected_gravity=np.asarray(projected_gravity, dtype=np.float32),
            dof_pos=np.asarray(dof_pos, dtype=np.float32),
            dof_vel=np.asarray(dof_vel, dtype=np.float32),
        )
        action = self.session.run([self.output_name], {"actor_obs": actor_obs})[0][0]
        action = np.clip(action.astype(np.float32), -self.clip_actions, self.clip_actions)
        target_q = self.default_dof_pos + action * self.action_scale

        self._update_history(pieces)
        self.last_action = action
        self.timer += 1
        return target_q.astype(np.float32), action


class AdamG1Retargeter:
    """Streaming G1 qpos -> Adam Pro qpos retargeter."""

    def __init__(
        self,
        package_root: str,
        g1_default_dof_pos: Sequence[float],
        g1_default_root_pos: Sequence[float],
        g1_default_root_quat_wxyz: Sequence[float],
        max_iter: int = 5,
        solver: str = "daqp",
        damping: float = 5e-1,
        retarget_every_n: int = 1,
        input_epsilon: float = 0.0,
        verbose: bool = False,
    ):
        self.package_root = Path(package_root).expanduser()
        if not self.package_root.exists():
            raise FileNotFoundError(f"robot_to_robot_retargeting path not found: {self.package_root}")
        if str(self.package_root) not in sys.path:
            sys.path.insert(0, str(self.package_root))

        try:
            from general_motion_retargeting import RobotToRobotRetargeting
        except ImportError as exc:
            raise ImportError(
                "Adam tracking mode requires robot_to_robot_retargeting and its dependencies. "
                f"Could not import from {self.package_root}."
            ) from exc

        self.g1_default_dof_pos = _as_float_array(
            g1_default_dof_pos, (29,), "ADAM_G1_DEFAULT_DOF_POS"
        )
        self.g1_default_root_pos = _as_float_array(
            g1_default_root_pos, (3,), "ADAM_G1_DEFAULT_ROOT_POS"
        )
        self.g1_default_root_quat_wxyz = _as_float_array(
            g1_default_root_quat_wxyz, (4,), "ADAM_G1_DEFAULT_ROOT_QUAT_WXYZ"
        )

        self.retargeter = RobotToRobotRetargeting(
            src_robot="unitree_g1",
            tgt_robot="pnd_adam_pro",
            solver=solver,
            damping=damping,
            verbose=verbose,
        )
        self.retargeter.retargeter.max_iter = int(max_iter)
        self.retarget_every_n = max(1, int(retarget_every_n))
        self.input_epsilon = float(input_epsilon)
        self.reset()

    def reset(self):
        self.retarget_counter = 0
        self.last_g1_dof_pos = self.g1_default_dof_pos.copy()
        self.last_adam_qpos = None

    def retarget(
        self,
        g1_dof_pos: Sequence[float] | None,
        g1_root_pos: Sequence[float] | None = None,
        g1_root_quat_wxyz: Sequence[float] | None = None,
    ) -> np.ndarray:
        if g1_dof_pos is None:
            g1_dof = self.last_g1_dof_pos
        else:
            g1_dof = _as_float_array(g1_dof_pos, (29,), "G1 decoder dof_pos")

        if (
            self.last_adam_qpos is not None
            and self.input_epsilon > 0.0
            and np.max(np.abs(g1_dof - self.last_g1_dof_pos)) <= self.input_epsilon
        ):
            return self.last_adam_qpos.copy()

        if self.last_adam_qpos is not None and self.retarget_counter % self.retarget_every_n != 0:
            self.retarget_counter += 1
            return self.last_adam_qpos.copy()

        self.last_g1_dof_pos = g1_dof.copy()

        root_pos = (
            self.g1_default_root_pos
            if g1_root_pos is None
            else _as_float_array(g1_root_pos, (3,), "G1 root position")
        )
        root_quat = (
            self.g1_default_root_quat_wxyz
            if g1_root_quat_wxyz is None
            else _as_float_array(g1_root_quat_wxyz, (4,), "G1 root quaternion")
        )
        g1_qpos = self.retargeter.make_source_qpos(root_pos, root_quat, g1_dof)
        self.last_adam_qpos = self.retargeter.retarget_qpos(g1_qpos).astype(np.float32)
        self.retarget_counter += 1
        return self.last_adam_qpos.copy()


class AdamTrackingReferenceBuilder:
    """Builds Adam tracking ONNX reference body-position tensors."""

    def __init__(
        self,
        adam_policy_joint_names: Sequence[str],
        fk_xml_path: str,
        source_xml_path: str,
        body_names: Sequence[str],
        extend_config: Sequence[dict],
        fk_body_names: Sequence[str] | None = None,
        fk_extend_config: Sequence[dict] | None = None,
        future_steps: int = 10,
    ):
        self.policy_joint_names = list(adam_policy_joint_names)
        self.future_steps = int(future_steps)
        self.model = mujoco.MjModel.from_xml_path(str(Path(fk_xml_path).expanduser()))
        self.data = mujoco.MjData(self.model)
        self.source_model = mujoco.MjModel.from_xml_path(str(Path(source_xml_path).expanduser()))
        fk_body_names = body_names if fk_body_names is None else fk_body_names
        fk_extend_config = extend_config if fk_extend_config is None else fk_extend_config

        self.qpos_ids = []
        self.source_qpos_ids = []
        for name in self.policy_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Adam FK model is missing policy joint '{name}'")
            self.qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            source_joint_id = mujoco.mj_name2id(
                self.source_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if source_joint_id == -1:
                raise ValueError(f"Adam source model is missing policy joint '{name}'")
            self.source_qpos_ids.append(int(self.source_model.jnt_qposadr[source_joint_id]))
        self.qpos_ids = np.asarray(self.qpos_ids, dtype=int)
        self.source_qpos_ids = np.asarray(self.source_qpos_ids, dtype=int)

        self.body_ids = []
        for name in fk_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                raise ValueError(f"Adam FK model is missing body '{name}'")
            self.body_ids.append(int(body_id))
        self.body_ids = np.asarray(self.body_ids, dtype=int)

        self.extend_parent_ids = []
        self.extend_pos = []
        self.extend_rot = []
        for item in fk_extend_config:
            parent_name = item["parent_name"]
            parent_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
            if parent_id == -1:
                raise ValueError(f"Adam FK model is missing extend parent '{parent_name}'")
            self.extend_parent_ids.append(int(parent_id))
            self.extend_pos.append(np.asarray(item["pos"], dtype=np.float32))
            rot_wxyz = np.asarray(item["rot"], dtype=np.float32)
            self.extend_rot.append(rot_wxyz[[1, 2, 3, 0]])
        self.extend_parent_ids = np.asarray(self.extend_parent_ids, dtype=int)
        self.extend_pos = np.asarray(self.extend_pos, dtype=np.float32)
        self.extend_rot = np.asarray(self.extend_rot, dtype=np.float32)
        self.num_markers = len(self.body_ids) + len(self.extend_parent_ids)

    def _make_fk_qpos(self, adam_qpos: np.ndarray) -> np.ndarray:
        adam_qpos = np.asarray(adam_qpos, dtype=np.float32)
        if adam_qpos.shape == (self.model.nq,):
            return adam_qpos.copy()
        if adam_qpos.shape == (self.source_model.nq,):
            fk_qpos = self.model.qpos0.copy().astype(np.float32)
            fk_qpos[:7] = adam_qpos[:7]
            fk_qpos[self.qpos_ids] = adam_qpos[self.source_qpos_ids]
            return fk_qpos
        raise ValueError(
            f"Expected Adam qpos length {self.model.nq} or {self.source_model.nq}, "
            f"got {adam_qpos.shape}"
        )

    def _body_positions_from_adam_qpos(self, adam_qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self._make_fk_qpos(adam_qpos)
        mujoco.mj_forward(self.model, self.data)
        body_pos = self.data.xpos[self.body_ids].copy().astype(np.float32)

        if len(self.extend_parent_ids) == 0:
            return body_pos

        parent_pos = self.data.xpos[self.extend_parent_ids].copy().astype(np.float32)
        parent_rot_xyzw = self.data.xquat[self.extend_parent_ids][:, [1, 2, 3, 0]].copy()
        rotated = np.stack(
            [
                _quat_rotate_xyzw(parent_rot_xyzw[i], self.extend_pos[i])
                for i in range(len(self.extend_parent_ids))
            ],
            axis=0,
        )
        # Preserve the HumanoidVLA online implementation's ordering behavior.
        extend_pos = np.stack(
            [
                _quat_rotate_xyzw(self.extend_rot[i], rotated[i]) + parent_pos[i]
                for i in range(len(self.extend_parent_ids))
            ],
            axis=0,
        )
        return np.concatenate([body_pos, extend_pos], axis=0).astype(np.float32)

    def build_repeated_reference(self, adam_qpos: np.ndarray) -> dict[str, np.ndarray]:
        markers = self._body_positions_from_adam_qpos(adam_qpos)
        ref = np.repeat(markers[None, :, :], self.future_steps, axis=0)
        return {
            "link_location": ref.reshape(-1).astype(np.float32),
            "joint_position": np.zeros((self.future_steps, 31), dtype=np.float32).reshape(-1),
        }


class AdamTrackingOnnxPolicy:
    """Runs Adam's tracking ONNX policy using G1 decoder commands as reference."""

    CURRENT_ORDER = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
    HISTORY_ORDER = [
        "actions",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "history_body_pos",
        "history_dif_body_pos",
        "projected_gravity",
    ]
    FUTURE_ORDER = ["future_dif_local_rigid_body_pos", "future_local_ref_body_pos_extend"]

    def __init__(
        self,
        onnx_path: str,
        joint_names: Sequence[str],
        default_dof_pos: Sequence[float],
        kp: Sequence[float],
        effort_limit: Sequence[float],
        retargeter: AdamG1Retargeter,
        reference_builder: AdamTrackingReferenceBuilder,
        control_dt: float = 0.02,
        command: Sequence[float] | None = None,
        history_len: int = 10,
        future_steps: int = 10,
        clip_observations: float = 100.0,
        clip_actions: float = 100.0,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Adam Pro tracking mode requires onnxruntime. Install it in .venv_sim with "
                "`pip install onnxruntime`."
            ) from exc

        self.onnx_path = Path(onnx_path).expanduser()
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Adam tracking ONNX policy not found: {self.onnx_path}")

        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_shapes = {
            inp.name: [dim if isinstance(dim, int) else -1 for dim in inp.shape]
            for inp in self.session.get_inputs()
        }
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.policy_kind = self._detect_policy_kind()

        self.joint_names = list(joint_names)
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float32)
        self.kp = np.asarray(kp, dtype=np.float32)
        self.effort_limit = np.asarray(effort_limit, dtype=np.float32)
        self.num_dof = len(self.joint_names)
        if self.default_dof_pos.shape != (self.num_dof,):
            raise ValueError("default_dof_pos must match Adam policy joint count")
        if self.kp.shape != (self.num_dof,):
            raise ValueError("kp must match Adam policy joint count")
        if self.effort_limit.shape != (self.num_dof,):
            raise ValueError("effort_limit must match Adam policy joint count")

        self.action_scale = 0.25 * self.effort_limit / self.kp
        self.retargeter = retargeter
        self.reference_builder = reference_builder
        self.control_dt = float(control_dt)
        self.command = (
            np.zeros(3, dtype=np.float32)
            if command is None
            else _as_float_array(command, (3,), "Adam command")
        )
        self.history_len = int(history_len)
        self.future_steps = int(future_steps)
        self.num_markers = self.reference_builder.num_markers
        self.marker_dim = self.num_markers * 3
        self.clip_observations = float(clip_observations)
        self.clip_actions = float(clip_actions)
        self.reset()

    def _detect_policy_kind(self) -> str:
        if set(self.input_names) == {
            "actor_obs_current",
            "actor_obs_past",
            "actor_obs_future",
        }:
            shapes = self.input_shapes
            expected = {
                "actor_obs_current": [1, 1, 75],
                "actor_obs_past": [1, 10, 249],
                "actor_obs_future": [1, 10, 174],
            }
            for name, shape in expected.items():
                if shapes[name][-len(shape) :] != shape:
                    raise ValueError(
                        f"Unexpected Adam tracking transformer shape for {name}: "
                        f"{shapes[name]}, expected {shape}"
                    )
            return "tracking_transformer"
        if self.input_names == ["actor_obs"] and self.input_shapes["actor_obs"][-1] == 2289:
            raise ValueError(
                "Adam flat tracking ONNX (`actor_obs [1, 2289]`) uses the older "
                "non-transformer observation layout and is not wired into this "
                "online G1-reference adapter yet. Use the transformer tracking "
                "model such as model_126000.onnx."
            )
        raise ValueError(
            "Adam tracking mode expects transformer inputs "
            "`actor_obs_current/past/future` or flat `actor_obs [1, 2289]`; "
            f"got inputs {self.input_shapes} from {self.onnx_path}."
        )

    def reset(self):
        self.retargeter.reset()
        self.timer = 0
        self.last_action = np.zeros(self.num_dof, dtype=np.float32)
        self.last_adam_ref_qpos = None
        self.last_reference_body_positions = None
        self.hist = {
            "actions": deque(maxlen=self.history_len),
            "base_ang_vel": deque(maxlen=self.history_len),
            "dof_pos": deque(maxlen=self.history_len),
            "dof_vel": deque(maxlen=self.history_len),
            "history_body_pos": deque(maxlen=self.history_len),
            "history_dif_body_pos": deque(maxlen=self.history_len),
            "projected_gravity": deque(maxlen=self.history_len),
        }
        zeros = {
            "actions": np.zeros(self.num_dof, dtype=np.float32),
            "base_ang_vel": np.zeros(3, dtype=np.float32),
            "dof_pos": np.zeros(self.num_dof, dtype=np.float32),
            "dof_vel": np.zeros(self.num_dof, dtype=np.float32),
            "history_body_pos": np.zeros(self.marker_dim, dtype=np.float32),
            "history_dif_body_pos": np.zeros(self.marker_dim, dtype=np.float32),
            "projected_gravity": np.zeros(3, dtype=np.float32),
        }
        for key, value in zeros.items():
            for _ in range(self.history_len):
                self.hist[key].append(value.copy())

    def set_command(self, command: Sequence[float]):
        self.command = _as_float_array(command, (3,), "Adam command")

    def get_reference_body_positions(self) -> np.ndarray | None:
        """Return the world-space reference markers used by the latest policy tick."""
        if self.last_reference_body_positions is None:
            return None
        return self.last_reference_body_positions.copy()

    def _localize_body_positions(
        self,
        base_quat_xyzw: np.ndarray,
        root_pos: np.ndarray,
        body_positions: np.ndarray,
    ) -> np.ndarray:
        heading_inv = _quat_heading_inv_xyzw(base_quat_xyzw)
        flat = (body_positions.reshape(-1, 3) - root_pos.reshape(1, 3)).astype(np.float32)
        return Rotation.from_quat(heading_inv).apply(flat).reshape(-1).astype(np.float32)

    def _history_actor(
        self,
        root_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        current_body_positions: np.ndarray,
    ) -> np.ndarray:
        pieces = []
        heading_inv = _quat_heading_inv_xyzw(base_quat_xyzw)
        heading_rot = Rotation.from_quat(heading_inv)
        for key in self.HISTORY_ORDER:
            hist = np.stack(list(self.hist[key]), axis=0)
            if key == "history_dif_body_pos":
                body_pos = hist.reshape(self.history_len, self.num_markers, 3)
                dif = body_pos - current_body_positions.reshape(1, self.num_markers, 3)
                hist = heading_rot.apply(dif.reshape(-1, 3)).reshape(self.history_len, -1)
            elif key == "history_body_pos":
                body_pos = hist.reshape(-1, 3)
                local = heading_rot.apply(body_pos - root_pos.reshape(1, 3))
                hist = local.reshape(self.history_len, -1)
            pieces.append(hist)
        return np.concatenate(pieces, axis=1).astype(np.float32)

    def _build_obs(
        self,
        root_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        current_body_positions: np.ndarray,
        future_ref_positions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        rotation = Rotation.from_quat(base_quat_xyzw)
        projected_gravity = rotation.apply(np.array([0.0, 0.0, -1.0]), inverse=True)

        current = {
            "actions": self.last_action.astype(np.float32),
            "base_ang_vel": (np.asarray(base_ang_vel, dtype=np.float32) * 0.25),
            "dof_pos": (np.asarray(dof_pos, dtype=np.float32) - self.default_dof_pos),
            "dof_vel": (np.asarray(dof_vel, dtype=np.float32) * 0.05),
            "projected_gravity": np.asarray(projected_gravity, dtype=np.float32),
        }

        future_local = []
        future_dif = []
        heading_inv = _quat_heading_inv_xyzw(base_quat_xyzw)
        heading_rot = Rotation.from_quat(heading_inv)
        for step in range(self.future_steps):
            ref_pos = future_ref_positions[step]
            future_local.append(
                heading_rot.apply(ref_pos.reshape(-1, 3) - root_pos.reshape(1, 3)).reshape(-1)
            )
            future_dif.append(
                heading_rot.apply(
                    ref_pos.reshape(-1, 3) - current_body_positions.reshape(-1, 3)
                ).reshape(-1)
            )
        future = {
            "future_dif_local_rigid_body_pos": np.asarray(future_dif, dtype=np.float32),
            "future_local_ref_body_pos_extend": np.asarray(future_local, dtype=np.float32),
        }

        history_entry = {
            **current,
            "history_body_pos": current_body_positions.reshape(-1).astype(np.float32),
            "history_dif_body_pos": current_body_positions.reshape(-1).astype(np.float32),
        }
        history_actor = self._history_actor(root_pos, base_quat_xyzw, current_body_positions)

        current_obs = np.concatenate([current[key].reshape(-1) for key in self.CURRENT_ORDER])
        if current_obs.shape != (75,):
            raise ValueError(f"Adam tracking current obs should be 75, got {current_obs.shape}")
        past_obs = history_actor
        future_obs = np.concatenate([future[key] for key in self.FUTURE_ORDER], axis=1)

        if self.policy_kind == "tracking_transformer":
            obs = {
                "actor_obs_current": np.clip(
                    current_obs.reshape(1, 1, -1),
                    -self.clip_observations,
                    self.clip_observations,
                ).astype(np.float32),
                "actor_obs_past": np.clip(
                    past_obs.reshape(1, self.history_len, -1),
                    -self.clip_observations,
                    self.clip_observations,
                ).astype(np.float32),
                "actor_obs_future": np.clip(
                    future_obs.reshape(1, self.future_steps, -1),
                    -self.clip_observations,
                    self.clip_observations,
                ).astype(np.float32),
            }
        else:
            obs = {
                "actor_obs": np.clip(
                    np.concatenate([current_obs, future_obs.reshape(-1), past_obs.reshape(-1)]).reshape(
                        1, -1
                    ),
                    -self.clip_observations,
                    self.clip_observations,
                ).astype(np.float32)
            }
        return obs, history_entry

    def _update_history(self, history_entry: dict[str, np.ndarray]):
        for key in self.HISTORY_ORDER:
            self.hist[key].appendleft(history_entry[key].astype(np.float32))

    def compute_target(
        self,
        root_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        current_body_positions: np.ndarray,
        g1_reference_dof_pos: Sequence[float] | None,
        g1_reference_root_pos: Sequence[float] | None = None,
        g1_reference_root_quat_wxyz: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        adam_ref_qpos = self.retargeter.retarget(
            g1_reference_dof_pos,
            g1_root_pos=g1_reference_root_pos,
            g1_root_quat_wxyz=g1_reference_root_quat_wxyz,
        )
        self.last_adam_ref_qpos = adam_ref_qpos
        ref_data = self.reference_builder.build_repeated_reference(adam_ref_qpos)
        future_ref_positions = ref_data["link_location"].reshape(
            self.future_steps, self.num_markers, 3
        )
        self.last_reference_body_positions = future_ref_positions[0].copy()

        obs, history_entry = self._build_obs(
            root_pos=np.asarray(root_pos, dtype=np.float32),
            base_quat_xyzw=np.asarray(base_quat_xyzw, dtype=np.float32),
            base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
            dof_pos=np.asarray(dof_pos, dtype=np.float32),
            dof_vel=np.asarray(dof_vel, dtype=np.float32),
            current_body_positions=np.asarray(current_body_positions, dtype=np.float32),
            future_ref_positions=future_ref_positions,
        )
        action = self.session.run([self.output_name], obs)[0][0]
        action = np.clip(action.astype(np.float32), -self.clip_actions, self.clip_actions)
        target_q = self.default_dof_pos + action * self.action_scale

        self._update_history(history_entry)
        self.last_action = action
        self.timer += 1
        return target_q.astype(np.float32), action
