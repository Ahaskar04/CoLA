from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from mink import SO3
from utils import setup_dual_arm_ik, compensate_gravity, check_gripper_box_contact, check_gripper_box_contact_right

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent  
_XML = _PROJECT_ROOT / "environments" / "handover_2arm" / "scene.xml"

setup = setup_dual_arm_ik(_XML)

box_x_range = (-0.12, 0.12)
box_y_range = (-0.10, 0.10)


def render_camera(model, data, renderer, camera_name):
    """Render one frame from a named camera."""
    renderer.update_scene(data, camera=camera_name)
    return renderer.render()


class EpisodeData:
    """Container for one episode's recorded data."""

    def __init__(self):
        self.image_overhead = []
        self.image_wrist_a = []
        self.image_wrist_b = []
        self.state_a = []
        self.state_b = []
        self.action_a = []
        self.action_b = []
        self.box_pos = []
        self.phase = []

    def add_step(self, image_overhead, image_wrist_a, image_wrist_b,
                 state_a, state_b, action_a, action_b, box_pos, phase):
        self.image_overhead.append(image_overhead)
        self.image_wrist_a.append(image_wrist_a)
        self.image_wrist_b.append(image_wrist_b)
        self.state_a.append(state_a)
        self.state_b.append(state_b)
        self.action_a.append(action_a)
        self.action_b.append(action_b)
        self.box_pos.append(box_pos)
        self.phase.append(phase)

    def to_dict(self):
        return {
            "image_overhead": np.array(self.image_overhead),
            "image_wrist_a": np.array(self.image_wrist_a),
            "image_wrist_b": np.array(self.image_wrist_b),
            "state_a": np.array(self.state_a),
            "state_b": np.array(self.state_b),
            "action_a": np.array(self.action_a),
            "action_b": np.array(self.action_b),
            "box_pos": np.array(self.box_pos),
            "phase": np.array(self.phase, dtype="S20"), 
        }


def run_episode(setup, box_x_range, box_y_range, max_steps=2000,
                 record_training_data=True, record_every_n_steps=10, 
                 record_review_video=True,
                 review_video_every_n_steps=10):
    """
    Run one full pick-and-handover episode, headless (no live viewer).

    Returns:
        episode_data: EpisodeData object (or None if record_training_data=False)
        success: bool, True if the episode reached the "done" phase
        review_frames: list of rendered frames for a human-review video (or None)
    """
    model = setup["model"]
    data = setup["data"]
    configuration = setup["configuration"]
    tasks = setup["tasks"]
    left_ee_task = setup["left_ee_task"]
    right_ee_task = setup["right_ee_task"]
    posture_task = setup["posture_task"]
    left_dof_ids = setup["left_dof_ids"]
    left_actuator_ids = setup["left_actuator_ids"]
    right_dof_ids = setup["right_dof_ids"]
    right_actuator_ids = setup["right_actuator_ids"]
    limits = setup["limits"]
    solver = setup["solver"]
    left_mocap_id = setup["left_mocap_id"]
    right_mocap_id = setup["right_mocap_id"]
    left_subtree_id = setup["left_subtree_id"]
    right_subtree_id = setup["right_subtree_id"]
    gripper_open = setup["GRIPPER_OPEN"]
    gripper_closed = setup["GRIPPER_CLOSED"]
    left_gripper_actuator_id = setup["left_gripper_actuator_id"]
    right_gripper_actuator_id = setup["right_gripper_actuator_id"]
    GRASP_ORIENTATION = setup["GRASP_ORIENTATION"]
    LIFT_HEIGHT = setup["LIFT_HEIGHT"]
    box_joint_id = setup["box_joint_id"]
    box_spawn_height = setup["box_spawn_height"]

    dt = 1.0 / 200.0
    APPROACH_HEIGHT_OFFSET = 0.10
    POS_THRESHOLD = 0.02
    B_APPROACH_HEIGHT_OFFSET = 0.08
    right_target_offset = np.array([0.03, 0.0, 0.0])

    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    posture_task.set_target_from_configuration(configuration)

    rand_x = np.random.uniform(*box_x_range)
    rand_y = np.random.uniform(*box_y_range)
    data.qpos[box_joint_id: box_joint_id + 3] = [rand_x, rand_y, box_spawn_height]
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
    mujoco.mj_forward(model, data)
    
    left_neutral_pos = data.mocap_pos[left_mocap_id].copy()
    left_neutral_quat = data.mocap_quat[left_mocap_id].copy()

    right_neutral_pos = data.mocap_pos[right_mocap_id].copy()
    right_neutral_quat = data.mocap_quat[right_mocap_id].copy()

    phase = "approach"
    gripper_ctrl = gripper_closed
    right_gripper_ctrl = gripper_closed
    right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    episode_data = EpisodeData() if record_training_data else None
    review_frames = [] if record_review_video else None
    renderer = mujoco.Renderer(model, 256, 256) if (record_training_data or record_review_video) else None

    step_count = 0
    target_pos = data.site('left/gripper').xpos.copy()  
    target_pos_b = data.site('right/gripper').xpos.copy()
    A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0]) 

    while phase != "done" and step_count < max_steps:
        box_pos = data.joint('middle_box_joint').qpos[:3]

        if phase == "approach":
            target_pos = box_pos + A_GRASP_OFFSET + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])

        elif phase == "open_gripper":
            gripper_ctrl = gripper_open

        elif phase == "descend":
            target_pos = box_pos.copy() + A_GRASP_OFFSET

        elif phase == "grasp":
            gripper_ctrl = gripper_closed

        elif phase == "lift":
            target_pos = np.array([box_pos[0] + A_GRASP_OFFSET[0], box_pos[1], LIFT_HEIGHT])

        elif phase == "move_arm_B":
            target_pos_b = target_pos + right_target_offset + np.array([0.0, 0.0, B_APPROACH_HEIGHT_OFFSET])
            data.mocap_pos[right_mocap_id] = target_pos_b
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
            right_gripper_ctrl = gripper_open

        elif phase == "approach_b":
            target_pos_b = target_pos + right_target_offset
            data.mocap_pos[right_mocap_id] = target_pos_b
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

        elif phase == "grip_b":
            right_gripper_ctrl = gripper_closed

        elif phase == "release_a":
            gripper_ctrl = gripper_open

        elif phase == "retract_b":
            data.mocap_pos[right_mocap_id] = right_neutral_pos
            data.mocap_quat[right_mocap_id] = right_neutral_quat
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

        if phase != "retract_b":
            target_quat = GRASP_ORIENTATION.wxyz
            data.mocap_pos[left_mocap_id] = target_pos
            data.mocap_quat[left_mocap_id] = target_quat
            left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

        vel = mink.solve_ik(configuration, tasks, dt, limits=limits, solver=solver, damping=1e-5)
        configuration.integrate_inplace(vel, dt)

        gripper_qpos = data.joint('left/left_finger').qpos[0]
        gripper_is_open = abs(gripper_qpos - gripper_open) < 0.002
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]

        current_gripper_pos = data.site('left/gripper').xpos
        dist_to_target = np.linalg.norm(current_gripper_pos - target_pos)
        right_gripper_pos = data.site('right/gripper').xpos
        dist_right_to_target = np.linalg.norm(right_gripper_pos - target_pos_b)

        if phase == "approach" and dist_to_target < POS_THRESHOLD:
            phase = "open_gripper"
        elif phase == "open_gripper" and gripper_is_open:
            phase = "descend"
        elif phase == "descend" and dist_to_target < POS_THRESHOLD:
            phase = "grasp"
        elif phase == "grasp" and check_gripper_box_contact(model, data):
            phase = "lift"
        elif phase == "lift" and dist_to_target < POS_THRESHOLD:
            phase = "move_arm_B"
        elif phase == "move_arm_B" and dist_right_to_target < POS_THRESHOLD:
            phase = "approach_b"
        elif phase == "approach_b" and dist_right_to_target < POS_THRESHOLD:
            phase = "grip_b"
        elif phase == "grip_b" and check_gripper_box_contact_right(model, data):
            phase = "release_a"
        elif phase == "release_a" and gripper_is_open:
            phase = "retract_b"
        elif phase == "retract_b" and np.linalg.norm(right_gripper_pos - right_neutral_pos) < POS_THRESHOLD:
            phase = "done"

        data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
        data.ctrl[left_gripper_actuator_id] = gripper_ctrl
        data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
        data.ctrl[right_gripper_actuator_id] = right_gripper_ctrl

        compensate_gravity(model, data, [left_subtree_id, right_subtree_id])
        mujoco.mj_step(model, data)

        # --- capture training data every step ---
        if record_training_data and step_count % record_every_n_steps == 0:
            state_a = np.concatenate([data.qpos[left_dof_ids], [gripper_qpos]])
            state_b = np.concatenate([data.qpos[right_dof_ids], [right_gripper_qpos]])
            action_a = np.concatenate([configuration.q[left_dof_ids], [gripper_ctrl]])
            action_b = np.concatenate([configuration.q[right_dof_ids], [right_gripper_ctrl]])

            episode_data.add_step(
                image_overhead=render_camera(model, data, renderer, "overhead_cam"),
                image_wrist_a=render_camera(model, data, renderer, "wrist_cam_left"),
                image_wrist_b=render_camera(model, data, renderer, "wrist_cam_right"),
                state_a=state_a,
                state_b=state_b,
                action_a=action_a,
                action_b=action_b,
                box_pos=box_pos.copy(),
                phase=phase,
            )

        # --- capture review video, sparsely (not every step, for speed) ---
        if record_review_video and step_count % review_video_every_n_steps == 0:
            renderer.update_scene(data, camera="overhead_cam")
            review_frames.append(renderer.render())

        step_count += 1

    if renderer is not None:
        renderer.close()

    success = (phase == "done")
    return episode_data, success, review_frames

       