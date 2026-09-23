"""
Pretrained-Octo rollout evaluation on the ALOHA handover scene.

Purpose: answer one question. Can a frozen, pretrained Octo pick the box up on
this robot, in this simulator, at all? The headline number is a_lift_rate.
Everything else is scored the same way cola_eval_aloha.py scores it, so the
rows line up, but do not expect the handover criteria to fire -- a single Octo
policy has no partner channel and no reason to time a transfer.

Scoring mirrors cola_eval_aloha.py exactly: same LIFT_Z, HOLD_STEPS,
SETTLE_STEPS, DROP_STEPS, same two-phase transfer/hold/return criteria, same
video labels, same summary keys.

------------------------------------------------------------------------------
WHAT THIS NUMBER CAN AND CANNOT TELL YOU
------------------------------------------------------------------------------
Octo emits 7-d DELTA END-EFFECTOR actions (dx dy dz, droll dpitch dyaw,
gripper) normalized to a source dataset. This robot takes ABSOLUTE joint
position targets. Three lossy conversions sit between them, all here:

  1. Unnormalization uses --octo-dataset's action statistics.
  2. Delta position is converted to joint targets by IK against the scripted
     expert's mink rig. ORIENTATION DELTAS ARE DISCARDED -- the wrist is held
     at GRASP_ORIENTATION throughout. A policy that wanted to rotate cannot.
  3. Rate. Octo's source data is ~5 Hz; this loop is 50 Hz. --action-scale
     exists to absorb that and is UNCALIBRATED.

So a LOW score is ambiguous: it is consistent with Octo being unable to do the
task here, and equally consistent with any of the three conversions being
wrong. A HIGH score is unambiguous and is the useful outcome -- it would mean
the frozen backbone really does carry the pick, and that an adapter only has to
add coordination.

Run --probe-actions FIRST. It reports raw Octo delta magnitudes with no physics
and no IK, which separates "the model outputs nothing useful" from "my bridge
is wrong" before you spend an hour on rollouts.
------------------------------------------------------------------------------

Usage:
    # step 0: are the raw actions even sane?
    python eval_octo_pickup.py --probe-actions --episodes 5

    # step 1: does it pick anything up?
    python eval_octo_pickup.py --episodes 50 --criterion transfer

    # step 2: if a_lift_rate is 0 everywhere, sweep the one free parameter
    for s in 0.02 0.05 0.1 0.25 0.5; do
        python eval_octo_pickup.py --episodes 20 --action-scale $s --no-videos \
            --results-path logs/octo_scale_$s.json
    done
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

# scene_fov58.xml, NOT scene.xml. The wrist cameras in aloha.xml were later
# widened from focal 1.93e-3 (fovy 58.0 deg) to 1.2e-3 (83.4 deg), but every
# cached demo image and extracted feature was rendered at 58.0 deg. Evaluating
# against the widened camera shows the policy a visual distribution it never
# trained on. This scene pins the original FOV so training and rollout match.
SCENE_XML = '/home/users/ntu/ahaskar0/CoLA/environments/handover_2arm/scene_fov58.xml'
EXPERT_DIR = '/home/users/ntu/ahaskar0/CoLA/data_collection/handover_2arm'

# Physics steps per control step. Matches cola_eval_aloha.py, which matches
# collect_demos.py record_every_n_steps=10.
CONTROL_DECIMATION = 10

BOX_X_RANGE = (-0.12, 0.12)
BOX_Y_RANGE = (-0.10, 0.10)
BOX_SPAWN_HEIGHT = 0.03
HO_Y_RANGE = 0.10
HO_Z_RANGE = 0.06

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

VIDEO_CAMERA = 'teleoperator_pov'

GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002

# --------------------------------------------------------------------------
# Criterion constants. Identical to cola_eval_aloha.py and expert_replay.py.
# --------------------------------------------------------------------------
LIFT_Z = 0.10
HOLD_STEPS = 5
SETTLE_STEPS = 20
DROP_STEPS = 3
HOME_TOL_RAD = 0.10
HOME_HOLD_STEPS = 5
GRIPPER_OPEN_FRAC = 0.6

CRITERIA = ('transfer', 'hold', 'return')

# --------------------------------------------------------------------------
# Octo
# --------------------------------------------------------------------------
OCTO_CHECKPOINTS = {
    'octo-small': 'hf://rail-berkeley/octo-small-1.5',
    'octo-base': 'hf://rail-berkeley/octo-base-1.5',
}
# Octo is language-conditioned, so this is the only task signal it receives.
# Deliberately NOT "coordinate with partner": that phrase describes the
# coordination, not the manipulation, and this script is testing manipulation.
INSTRUCTION_A = 'pick up the red box'
INSTRUCTION_B = 'take the red box from the other arm'
# mink iterations per commanded delta. More is closer tracking, and slower.
IK_ITERS = 40
IK_DT = 1.0 / 200.0
# Chunk length to execute open-loop between Octo queries. Matches CoLA's
# CHUNK_SIZE so the control cadence is comparable; Octo's own horizon is
# usually longer and gets truncated to this.
CHUNK_SIZE = 10


def build_scene(xml_path: str) -> Dict:
    """Load the scene and cache every id the rollout needs.

    Identical to cola_eval_aloha.build_scene, plus the gripper site name per
    arm, which the IK bridge needs.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def arm(prefix):
        return {
            'prefix': prefix,
            'actuators': np.array([model.actuator(f'{prefix}/{n}').id for n in ARM_JOINTS]),
            'gripper_actuator': model.actuator(f'{prefix}/gripper').id,
            'subtree': model.body(f'{prefix}/base_link').id,
            'qadr': np.array([model.joint(f'{prefix}/{n}').qposadr[0] for n in ARM_JOINTS]),
            'finger_qadr': model.joint(f'{prefix}/left_finger').qposadr[0],
            # Both finger pads. CoLA's check_gripper_box_contact_right lists
            # right/right_g* twice and omits right/left_g*, so it misses
            # contacts on one pad; this does not.
            'finger_geoms': {
                model.geom(f'{prefix}/{side}_g{i}').id
                for side in ('left', 'right') for i in range(3)
            },
            'site': f'{prefix}/gripper',
        }

    a, b = arm('left'), arm('right')
    neutral_key = model.key('neutral_pose').id

    mujoco.mj_resetDataKeyframe(model, data, neutral_key)
    home_qpos_a = data.qpos[a['qadr']].copy()
    home_qpos_b = data.qpos[b['qadr']].copy()

    return {
        'model': model,
        'data': data,
        'a': a,
        'b': b,
        'box_qposadr': model.joint('middle_box_joint').qposadr[0],
        'box_geom': model.geom('middle_box_geom').id,
        'neutral_key': neutral_key,
        'home_qpos_a': home_qpos_a,
        'home_qpos_b': home_qpos_b,
    }


def compensate_gravity(model, data, subtree_ids):
    """Cancel gravity on the arm subtrees, as collect_demos.py did."""
    qfrc_applied = data.qfrc_applied
    qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


def touching_box(data, box_geom: int, finger_geoms: set) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == box_geom and c.geom2 in finger_geoms) or \
           (c.geom2 == box_geom and c.geom1 in finger_geoms):
            return True
    return False


_HO_SETUP = None


def _expert_setup(scene_xml: str):
    """Lazily build the scripted expert's mink IK rig, reused across episodes.

    Needed for two things here: constructing the handover-only start state, and
    converting Octo's delta end-effector actions into joint targets. Both use
    the expert's own rig rather than a second implementation that could drift.
    """
    global _HO_SETUP
    if _HO_SETUP is None:
        import sys
        sys.path.insert(0, EXPERT_DIR)
        from utils import setup_dual_arm_ik
        _HO_SETUP = setup_dual_arm_ik(scene_xml)
    return _HO_SETUP


def reset_episode(scene, seed: int):
    """Neutral pose, box spawned on the table. The pickup test."""
    model, data = scene['model'], scene['data']
    rng = np.random.default_rng(seed)

    mujoco.mj_resetDataKeyframe(model, data, scene['neutral_key'])
    addr = scene['box_qposadr']
    data.qpos[addr:addr + 3] = [
        rng.uniform(*BOX_X_RANGE),
        rng.uniform(*BOX_Y_RANGE),
        BOX_SPAWN_HEIGHT,
    ]
    mujoco.mj_forward(model, data)


def reset_episode_handover(scene, seed: int, scene_xml: str) -> bool:
    """Start with arm A already holding the box, as the handover-only demos do.

    Mirrors scripted_policy.run_episode's handover_only branch. Returns False if
    the randomised reset shook the box loose, so the caller can skip rather than
    score an episode that began with an empty gripper.
    """
    import mink
    su = _expert_setup(scene_xml)
    from utils import check_gripper_box_contact
    m, d = su['model'], su['data']
    cfg = su['configuration']
    rng = np.random.default_rng(seed)

    mujoco.mj_resetDataKeyframe(m, d, m.key('handover_start').id)
    mujoco.mj_forward(m, d)
    grip0 = d.site('left/gripper').xpos.copy()
    offset = np.array([0.0,
                       rng.uniform(-HO_Y_RANGE, HO_Y_RANGE),
                       rng.uniform(-HO_Z_RANGE, HO_Z_RANGE)])
    goal = grip0 + offset

    cfg.update(d.qpos)
    su['posture_task'].set_target_from_configuration(cfg)
    d.mocap_pos[su['left_mocap_id']] = goal
    d.mocap_quat[su['left_mocap_id']] = su['GRASP_ORIENTATION'].wxyz
    su['left_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "left/target"))
    mink.move_mocap_to_frame(m, d, "right/target", "right/gripper", "site")
    su['right_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "right/target"))

    gc = su['GRIPPER_CLOSED']
    for _ in range(1500):
        vel = mink.solve_ik(cfg, su['tasks'], IK_DT, limits=su['limits'],
                            solver=su['solver'], damping=1e-5)
        cfg.integrate_inplace(vel, IK_DT)
        d.ctrl[su['left_actuator_ids']] = cfg.q[su['left_dof_ids']]
        d.ctrl[su['right_actuator_ids']] = cfg.q[su['right_dof_ids']]
        d.ctrl[su['left_gripper_actuator_id']] = gc
        d.ctrl[su['right_gripper_actuator_id']] = gc
        mujoco.mj_step(m, d)
        if np.linalg.norm(d.site('left/gripper').xpos - goal) < 0.005:
            break

    if not check_gripper_box_contact(m, d):
        return False

    scene['data'].qpos[:] = d.qpos
    scene['data'].qvel[:] = 0.0
    scene['data'].ctrl[:] = d.ctrl
    mujoco.mj_forward(scene['model'], scene['data'])
    return True


def apply_action(data, arm, action):
    """Six joint position targets plus one gripper command."""
    data.ctrl[arm['actuators']] = action[:6]
    data.ctrl[arm['gripper_actuator']] = action[6]


def hold_chunk(scene, arm, n=CHUNK_SIZE):
    """A chunk that commands the arm to stay exactly where it is.

    Used for arm B when --arm-b hold. Reading the CURRENT joint positions
    rather than the neutral keyframe means B holds wherever it happens to be,
    including a handover-only start pose.
    """
    q = scene['data'].qpos[arm['qadr']].copy()
    grip = float(scene['data'].ctrl[arm['gripper_actuator']])
    chunk = np.zeros((n, 7), dtype=np.float32)
    chunk[:, :6] = q
    chunk[:, 6] = grip
    return chunk


class OctoPolicy:
    """A pretrained Octo checkpoint driving one arm, via an IK bridge.

    Read the module docstring before trusting any number from this. The short
    version: Octo speaks delta end-effector, this robot speaks absolute joint
    targets, and everything between them is uncalibrated.

    Observation handling: Octo consumes a short window of frames, not a single
    image, so the buffer is rebuilt from scratch at every reset(). Carrying
    frames across an episode boundary would put the transformer in a state no
    trajectory ever produced.
    """

    def __init__(self, checkpoint: str, dataset_name: str, scene_xml: str,
                 action_scale: float, instruction: str, camera: str,
                 gripper_open_high: bool = True, seed: int = 0,
                 wrist_orientation: str = 'grasp',
                 random_actions: bool = False):
        import jax
        from octo.model.octo_model import OctoModel

        self.jax = jax
        print(f'   loading {checkpoint} ...')
        self.model = OctoModel.load_pretrained(checkpoint)

        # Window size is not constant across Octo releases; read it off the
        # example batch the checkpoint ships with rather than hardcoding 2.
        self.window = int(
            self.model.example_batch['observation']['image_primary'].shape[1])

        # dataset_statistics is a dict-of-datasets on pretrained checkpoints and
        # flat on a single-dataset finetune. Support both; fail loudly with the
        # available keys rather than a bare KeyError.
        stats = self.model.dataset_statistics
        if 'action' in stats:
            self.action_stats = stats['action']
            self.stats_name = 'finetune'
        elif dataset_name in stats:
            self.action_stats = stats[dataset_name]['action']
            self.stats_name = dataset_name
        else:
            raise SystemExit(
                f'--octo-dataset {dataset_name!r} not in this checkpoint. '
                f'Available: {sorted(stats.keys())}')

        self.task = self.model.create_tasks(texts=[instruction])
        self.action_scale = float(action_scale)
        self.scene_xml = scene_xml
        self.camera = camera
        self.gripper_open_high = gripper_open_high
        self.wrist_orientation = wrist_orientation
        # Control condition: draw from Octo's OUTPUT MARGINAL instead of from
        # Octo. Same action distribution, zero visual or language
        # conditioning. If the score does not move, the conditioning is
        # buying nothing and the number is not measuring Octo's prior.
        self.random_actions = random_actions
        try:
            self.horizon = int(self.model.config['model']['heads']['action']
                               ['kwargs']['action_horizon'])
        except (KeyError, TypeError):
            self.horizon = 4
        # Set on the first bridge call after each reset, for 'initial' mode.
        self.hold_quat = None
        self.rng = jax.random.PRNGKey(seed)
        self.hist = None
        self.instruction = instruction
        self.n_real = 0
        # Raw delta magnitudes, accumulated across the run for the summary.
        self.delta_norms = []
        self.grip_values = []
        self.grip_chunk_values = []
        self.box_alignment = []

        print(f'   window {self.window} | action stats: {self.stats_name}')
        print(f'   instruction: {instruction!r}')
        print(f'   camera: {camera} | action scale {self.action_scale} (UNCALIBRATED)')
        print(f'   gripper: {"high=open" if gripper_open_high else "high=closed"}')
        print(f'   wrist held at: {wrist_orientation} | horizon {self.horizon}')
        if random_actions:
            print('   !! RANDOM ACTIONS: sampling the action marginal, '
                  'NOT querying Octo. This is the control condition.')

    def reset(self):
        self.hist = None
        self.hold_quat = None
        self.num_obs = 0

    def _push(self, image):
        self.num_obs += 1
        if self.hist is None:
            # Prime a fresh buffer by repeating the first frame. Standard, but
            # note it means the first few steps see no motion at all.
            self.hist = np.repeat(image[None, ...], self.window, axis=0)
        else:
            self.hist = np.concatenate([self.hist[1:], image[None, ...]], axis=0)

    def _observation(self):
        obs = {'image_primary': self.hist[None, ...].astype(np.uint8)}
        # Octo 1.5 renamed pad_mask -> timestep_pad_mask. Emit whichever this
        # checkpoint expects.
        key = ('timestep_pad_mask'
               if 'timestep_pad_mask' in self.model.example_batch['observation']
               else 'pad_mask')
        # Octo's own HistoryWrapper marks the repeated priming frames as
        # PADDING (gym_wrappers.stack_and_pad: mask[:horizon - min(num_obs,
        # horizon)] = 0), so the transformer masks them out. Sending all-True
        # instead claims two copies of one frame are two real consecutive
        # observations. Only affects the first query of each episode, but it
        # is a deviation from the reference for no reason.
        mask = np.ones((1, self.window), dtype=bool)
        mask[0, :self.window - min(self.num_obs, self.window)] = False
        obs[key] = mask
        return obs

    def raw_actions(self, scene, renderer):
        """Sample and unnormalize, with NO IK and NO physics. (horizon, 7)."""
        if self.random_actions:
            mean = np.asarray(self.action_stats['mean'], dtype=np.float64)
            std = np.asarray(self.action_stats['std'], dtype=np.float64)
            self.rng, sub_ = self.jax.random.split(self.rng)
            z = np.asarray(self.jax.random.normal(
                sub_, (self.horizon, len(mean))), dtype=np.float64)
            act = z * std + mean
            # Dim 6 carries mask=False in the bridge statistics, so it is
            # never unnormalized -- the head emits it directly in ~[0, 1].
            mask = np.asarray(self.action_stats.get(
                'mask', np.ones_like(mean, dtype=bool)))
            act[:, ~mask] = np.clip(z[:, ~mask] * 0.25 + 0.5, 0.0, 1.0)
            return act
        renderer.update_scene(scene['data'], camera=self.camera)
        self._push(renderer.render())

        self.rng, sub = self.jax.random.split(self.rng)
        actions = self.model.sample_actions(
            self._observation(), self.task,
            unnormalization_statistics=self.action_stats, rng=sub)
        actions = np.asarray(actions)
        # Shape varies by release: (batch, horizon, dim) or
        # (batch, window, horizon, dim). Keep the last two axes either way.
        return actions.reshape(-1, actions.shape[-2], actions.shape[-1])[0]

    def _delta_to_joint_targets(self, scene, arm, deltas):
        """Integrate delta xyz into absolute joint targets by IK.

        deltas is (K, 3) in metres, applied cumulatively from the arm's CURRENT
        gripper pose. Octo's ORIENTATION deltas are discarded: the rig tracks
        GRASP_ORIENTATION throughout, because the expert's task fixes wrist
        orientation and there is no validated mapping from Octo's rpy
        convention into this one. This is a real limitation of the baseline,
        not a detail -- a policy that wanted to rotate the wrist cannot.
        """
        import mink
        su = _expert_setup(self.scene_xml)
        m, d = su['model'], su['data']
        cfg = su['configuration']

        side = 'left' if arm['prefix'] == 'left' else 'right'
        mocap_id = su[f'{side}_mocap_id']
        dof_ids = su[f'{side}_dof_ids']
        ee_task = su[f'{side}_ee_task']

        # Sync the rig to the live scene, so deltas apply from where the arm
        # actually is rather than from where IK last left it.
        d.qpos[:] = scene['data'].qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        cfg.update(d.qpos)
        su['posture_task'].set_target_from_configuration(cfg)

        # Pin the arm we are NOT driving to wherever it currently is.
        # su['tasks'] holds BOTH FrameTasks, so solve_ik raises TargetNotSet
        # unless the idle one has a target -- and if it keeps a stale target
        # from an earlier chunk, the QP quietly drags that arm toward it.
        # Re-pinning every chunk fixes both.
        other = 'right' if side == 'left' else 'left'
        mink.move_mocap_to_frame(m, d, f'{other}/target', f'{other}/gripper', 'site')
        su[f'{other}_ee_task'].set_target(
            mink.SE3.from_mocap_name(m, d, f'{other}/target'))

        goal = d.site(f'{side}/gripper').xpos.copy()
        targets = np.zeros((len(deltas), 6), dtype=np.float32)

        # At neutral_pose the wrist sits ~78 deg away from GRASP_ORIENTATION,
        # so pinning to it slews the arm ~1.9 rad during the FIRST chunk of
        # every episode -- motion Octo never asked for. 'grasp' accepts that
        # transient because it is the orientation the box actually has to be
        # gripped from; 'initial' holds whatever the wrist starts at, which
        # commands no uncommanded motion but is unlikely to afford a grasp.
        # Either way Octo's own rotation deltas are discarded.
        if self.wrist_orientation == 'grasp':
            quat = su['GRASP_ORIENTATION'].wxyz
        else:
            if self.hold_quat is None:
                self.hold_quat = mink.SO3.from_matrix(
                    d.site(f'{side}/gripper').xmat.reshape(3, 3).copy()).wxyz
            quat = self.hold_quat

        for k, delta in enumerate(deltas):
            goal = goal + delta
            d.mocap_pos[mocap_id] = goal
            d.mocap_quat[mocap_id] = quat
            ee_task.set_target(mink.SE3.from_mocap_name(m, d, f'{side}/target'))
            for _ in range(IK_ITERS):
                vel = mink.solve_ik(cfg, su['tasks'], IK_DT, limits=su['limits'],
                                    solver=su['solver'], damping=1e-5)
                cfg.integrate_inplace(vel, IK_DT)
            targets[k] = cfg.q[dof_ids]

        return targets

    def act(self, scene, renderer, arm):
        """One chunk of (CHUNK_SIZE, 7) in raw ctrl units."""
        raw = self.raw_actions(scene, renderer)
        steps = raw[:CHUNK_SIZE]
        self.n_real = len(steps)
        if len(steps) < CHUNK_SIZE:
            # Octo's horizon (4 on octo-small-1.5) is shorter than CHUNK_SIZE,
            # so the tail of every chunk has to be invented. Columns 0-5 are
            # DELTAS, and holding position means padding them with ZERO.
            # Repeating the last delta -- which is what "hold the last action"
            # means in an ABSOLUTE action space -- keeps the arm travelling and
            # inflates the chunk's commanded displacement by 2.5x. Column 6 is
            # absolute, so the gripper genuinely is held.
            pad = np.zeros((CHUNK_SIZE - len(steps), steps.shape[1]),
                           dtype=steps.dtype)
            pad[:, 6] = steps[-1, 6]
            steps = np.concatenate([steps, pad], axis=0)

        # Diagnostics, so a zero score is attributable without a re-run. Only
        # the REAL actions are counted: the zero padding would otherwise drag
        # every delta statistic toward zero and hide the true action scale.
        real = steps[:self.n_real]
        self.delta_norms.extend(np.linalg.norm(real[:, :3], axis=1).tolist())
        self.grip_values.extend(real[:, 6].tolist())
        # One value per chunk, to measure how often the hand changes state.
        self.grip_chunk_values.append(float(real[-1, 6]))

        # Octo's dx/dy/dz are EGOCENTRIC: +x away from the robot base, +z up,
        # in the BASE frame of whatever arm produced the training data. The IK
        # bridge integrates onto a WORLD-frame gripper position, so the deltas
        # have to be rotated into world first. left/base_link happens to be
        # identity, but right/base_link is rotated 180 deg about z -- without
        # this, arm B is driven with its x and y inverted, i.e. backwards.
        R_base = scene['data'].xmat[arm['subtree']].reshape(3, 3)
        deltas = (steps[:, :3] * self.action_scale) @ R_base.T

        # Is Octo actually aiming at the box? This is the one diagnostic that
        # separates "undirected motion" from "directed at the wrong place":
        # cosine between the chunk's net commanded displacement and the
        # direction from the gripper to the box. ~0 means a random walk;
        # negative means systematically away; only clearly positive means the
        # policy is reaching.
        net = deltas.sum(axis=0)
        to_box = (scene['data'].qpos[scene['box_qposadr']:scene['box_qposadr'] + 3]
                  - scene['data'].site(arm['site']).xpos)
        dn, tn = np.linalg.norm(net), np.linalg.norm(to_box)
        if dn > 1e-9 and tn > 1e-9:
            self.box_alignment.append(float(net @ to_box / (dn * tn)))

        joints = self._delta_to_joint_targets(scene, arm, deltas)

        # Column 6 is Octo's gripper. Bridge-style data is ~1 open, ~0 closed;
        # other source datasets invert this. --gripper-inverted flips it.
        if self.gripper_open_high:
            grip = np.where(steps[:, 6] > 0.5, GRIPPER_OPEN, GRIPPER_CLOSED)
        else:
            grip = np.where(steps[:, 6] > 0.5, GRIPPER_CLOSED, GRIPPER_OPEN)

        chunk = np.zeros((CHUNK_SIZE, 7), dtype=np.float32)
        chunk[:, :6] = joints
        chunk[:, 6] = grip
        return chunk

    def diagnostics(self):
        if not self.delta_norms:
            return {}
        d = np.asarray(self.delta_norms)
        g = np.asarray(self.grip_values)
        # A grasp needs the hand to STAY closed. A gripper column that
        # re-rolls every chunk is as fatal as one that never moves, and the
        # frac-above-half number alone cannot tell the two apart.
        gc = np.asarray(self.grip_chunk_values) > 0.5
        flip = float((gc[1:] != gc[:-1]).mean()) if len(gc) > 1 else 0.0
        al = np.asarray(self.box_alignment) if self.box_alignment else np.zeros(0)
        return {
            'gripper_chunk_flip_rate': flip,
            'box_alignment_mean': float(al.mean()) if al.size else None,
            'box_alignment_frac_positive': float((al > 0).mean()) if al.size else None,
            'raw_delta_norm_mean': float(d.mean()),
            'raw_delta_norm_median': float(np.median(d)),
            'raw_delta_norm_p95': float(np.percentile(d, 95)),
            'raw_delta_norm_max': float(d.max()),
            'scaled_delta_mm_median': float(np.median(d) * self.action_scale * 1000),
            'gripper_raw_mean': float(g.mean()),
            'gripper_raw_frac_above_half': float((g > 0.5).mean()),
        }


def probe_actions(policy, scene, scene_xml, n_episodes, handover_only):
    """Sample Octo actions with no IK and no physics, and report their scale.

    The cheapest possible attribution step. If the deltas are ~0, or orders of
    magnitude off, no amount of rollout debugging will help and the problem is
    upstream of the bridge.
    """
    model = scene['model']
    renderer = mujoco.Renderer(model, 256, 256)
    all_raw = []

    for ep in range(n_episodes):
        if handover_only:
            if not reset_episode_handover(scene, ep, scene_xml):
                continue
        else:
            reset_episode(scene, seed=ep)
        policy.reset()
        # Query a few times from the start state without advancing physics.
        # The observation window fills with repeats of the same frame, so this
        # measures the action Octo commits to from a static scene.
        for _ in range(5):
            all_raw.append(policy.raw_actions(scene, renderer))

    renderer.close()
    if not all_raw:
        raise SystemExit(
            'probe collected no actions: every --handover-only reset dropped '
            'the box, so there was no valid start state to sample from.')
    raw = np.concatenate(all_raw, axis=0)
    pos, rot, grip = raw[:, :3], raw[:, 3:6], raw[:, 6]
    norms = np.linalg.norm(pos, axis=1)

    print('\n' + '=' * 60)
    print('RAW ACTION PROBE (no IK, no physics)')
    print('=' * 60)
    print(f'samples: {len(raw)}')
    print(f'position delta norm  mean {norms.mean():.5f}  median '
          f'{np.median(norms):.5f}  p95 {np.percentile(norms, 95):.5f}  '
          f'max {norms.max():.5f}   [metres per Octo step]')
    print(f'  per-axis mean |dx|,|dy|,|dz|: '
          f'{np.abs(pos).mean(axis=0).round(5).tolist()}')
    print(f'rotation delta       mean |drpy|: '
          f'{np.abs(rot).mean(axis=0).round(5).tolist()}   [DISCARDED by the bridge]')
    print(f'gripper column       mean {grip.mean():.3f}  min {grip.min():.3f}  '
          f'max {grip.max():.3f}  frac>0.5 {(grip > 0.5).mean():.3f}')
    print()
    print('How to read this:')
    print('  - norms near 0 (<1e-4): Octo is committing to no motion. The')
    print('    instruction, the camera, or the visual domain is the problem,')
    print('    not the IK bridge. Try --camera overhead_cam and a different')
    print('    --instruction before anything else.')
    print('  - gripper column stuck at one value: it will never open or close.')
    print('    Check --gripper-inverted, then check the source dataset.')
    print('  - norms of order 0.01-0.05 m: plausible for ~5 Hz source data.')
    print(f'    At --action-scale {policy.action_scale} that is '
          f'{np.median(norms) * policy.action_scale * 1000:.2f} mm per control')
    print('    step, or {:.1f} mm/s at 50 Hz.'.format(
        np.median(norms) * policy.action_scale * 1000 * 50))
    return raw


def evaluate(
    policy_a: OctoPolicy,
    policy_b: Optional[OctoPolicy] = None,
    n_episodes: int = 50,
    scene_xml: str = SCENE_XML,
    save_videos: bool = True,
    video_dir: str = 'octo_eval_videos',
    results_path: str = 'logs/octo_eval_results.json',
    max_control_steps: int = 350,
    video_camera: str = VIDEO_CAMERA,
    criterion: str = 'transfer',
    return_arm: str = 'b',
    handover_only: bool = False,
    arm_b_mode: str = 'hold',
    seed_offset: int = 0,
) -> Dict:
    assert criterion in CRITERIA, f'criterion must be one of {CRITERIA}'
    assert return_arm in ('a', 'b')
    assert arm_b_mode in ('hold', 'octo')
    # Octo is a SINGLE-ARM policy: one camera, one instruction, one 7-d output.
    # Two arms therefore need two independent instances. Sharing one would also
    # share its observation window, so each chunk would push two frames into a
    # window-2 buffer and both arms would be conditioned on one arm's camera.
    assert not (arm_b_mode == 'octo' and policy_b is None), \
        'arm_b_mode="octo" needs a second OctoPolicy for arm B'

    crit_desc = {
        'transfer': 'transfer only (legacy)',
        'hold': f'transfer + B keeps the box {SETTLE_STEPS} steps',
        'return': f'transfer + arm {return_arm.upper()} returns home, still holding',
    }[criterion]

    print(f'Success: {crit_desc}')
    print(f'Arm B: {"driven by Octo" if arm_b_mode == "octo" else "held still"}')

    videos_dir = Path(video_dir)
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print('\n2. Building ALOHA scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['a']['subtree'], scene['b']['subtree']]
    ret_arm = scene[return_arm]
    home_qpos = scene[f'home_qpos_{return_arm}']
    print('   scene ready')

    results = {
        'policy': 'octo-pretrained',
        'arm_b_mode': arm_b_mode,
        'action_scale': policy_a.action_scale,
        'action_stats': policy_a.stats_name,
        'arm_a': {'camera': policy_a.camera,
                  'instruction': policy_a.instruction},
        'arm_b': (None if policy_b is None else
                  {'camera': policy_b.camera,
                   'instruction': policy_b.instruction}),
        'criterion': {
            'name': criterion,
            'return_arm': return_arm,
            'lift_z': LIFT_Z,
            'hold_steps': HOLD_STEPS,
            'settle_steps': SETTLE_STEPS,
            'drop_steps': DROP_STEPS,
            'home_tol_rad': HOME_TOL_RAD,
            'home_hold_steps': HOME_HOLD_STEPS,
            'gripper_open_frac': GRIPPER_OPEN_FRAC,
            'max_control_steps': max_control_steps,
        },
        'episodes': [],
        'task_successes': 0,
        'transfer_count': 0,
        'drop_count': 0,
        'a_lifted_count': 0,
        'a_touched_count': 0,
        'b_touched_count': 0,
        'skipped': 0,
        # Recorded rather than inferred from the run tag: the start state is
        # not recoverable from the numbers, and a tag-name heuristic silently
        # mislabelled rows whose name did not end in _ho.
        'handover_only': bool(handover_only),
        # Which block of episode seeds this run drew. Episodes are seeded
        # seed_offset + ep_idx, exactly as cola_eval_aloha.py seeds them, so a
        # row evaluated at the same offset scores the identical start states.
        'seed_offset': int(seed_offset),
    }

    print(f'\n3. Running {n_episodes} episodes...')
    for ep_idx in tqdm(range(n_episodes), desc='Evaluating'):
        if handover_only:
            if not reset_episode_handover(scene, seed_offset + ep_idx, scene_xml):
                results['skipped'] += 1
                continue
        else:
            reset_episode(scene, seed=seed_offset + ep_idx)

        policy_a.reset()
        if policy_b is not None:
            policy_b.reset()

        frames = []
        a_lifted = False
        a_touched = False       # did A's fingers ever reach the box at all
        b_touched = False
        handover_run = 0
        transfer_done = False
        transfer_step = None
        settle_run = 0
        no_contact_run = 0
        home_run = 0
        dropped = False
        success = False
        control_step = 0

        max_box_z = 0.0
        max_handover_run = 0
        best_home_err = float('inf')
        # Closest A's gripper site ever got to the box. On a pickup test this
        # is the most informative single diagnostic: it separates "reached the
        # box but did not close" from "never went near it".
        min_a_box_dist = float('inf')

        while control_step < max_control_steps and not success and not dropped:
            chunk_a = policy_a.act(scene, renderer, scene['a'])
            if arm_b_mode == 'octo':
                chunk_b = policy_b.act(scene, renderer, scene['b'])
            else:
                chunk_b = hold_chunk(scene, scene['b'])

            # Execute what the policies actually predicted. A head trained at
            # action_horizon < CHUNK_SIZE (e.g. octo's native 4) returns a
            # shorter chunk; indexing past it would crash, and padding it would
            # replay a stale final pose. So re-query sooner instead. For every
            # existing checkpoint (horizon 50) this is CHUNK_SIZE, unchanged.
            n_exec = min(len(chunk_a), len(chunk_b), CHUNK_SIZE)
            for k in range(n_exec):
                if control_step >= max_control_steps:
                    break

                apply_action(data, scene['a'], chunk_a[k])
                apply_action(data, scene['b'], chunk_b[k])

                for _ in range(CONTROL_DECIMATION):
                    compensate_gravity(model, data, subtrees)
                    mujoco.mj_step(model, data)

                control_step += 1

                box_pos = data.qpos[scene['box_qposadr']:scene['box_qposadr'] + 3]
                box_z = float(box_pos[2])
                a_holds = touching_box(data, scene['box_geom'], scene['a']['finger_geoms'])
                b_holds = touching_box(data, scene['box_geom'], scene['b']['finger_geoms'])

                a_lifted |= a_holds and box_z > LIFT_Z
                a_touched |= a_holds
                b_touched |= b_holds
                max_box_z = max(max_box_z, box_z)
                min_a_box_dist = min(
                    min_a_box_dist,
                    float(np.linalg.norm(data.site(scene['a']['site']).xpos - box_pos)))

                a_open = (float(data.qpos[scene['a']['finger_qadr']])
                          > GRIPPER_OPEN * GRIPPER_OPEN_FRAC)
                b_has_box = b_holds and box_z > LIFT_Z

                if not transfer_done:
                    if b_has_box and not a_holds and a_open:
                        handover_run += 1
                    else:
                        handover_run = 0
                    max_handover_run = max(max_handover_run, handover_run)

                    if handover_run >= HOLD_STEPS:
                        transfer_done = True
                        transfer_step = control_step
                        if criterion == 'transfer':
                            success = True
                else:
                    no_contact_run = 0 if b_holds else no_contact_run + 1
                    if no_contact_run >= DROP_STEPS:
                        dropped = True
                        break

                    if criterion == 'return':
                        home_err = float(np.max(np.abs(
                            data.qpos[ret_arm['qadr']] - home_qpos)))
                        best_home_err = min(best_home_err, home_err)
                        home_run = home_run + 1 if home_err < HOME_TOL_RAD else 0
                        if home_run >= HOME_HOLD_STEPS:
                            success = True
                    else:
                        settle_run += 1
                        if settle_run >= SETTLE_STEPS:
                            success = True

                if save_videos:
                    renderer.update_scene(data, camera=video_camera)
                    frames.append(renderer.render())

                if success or dropped:
                    break

        if save_videos and frames:
            if success:
                label = 'SUCCESS'
            elif dropped:
                label = 'FAIL_DROPPED'
            elif transfer_done:
                label = 'FAIL_NORETURN'
            elif b_touched:
                label = 'FAIL_BTOUCH_NOXFER'
            elif a_lifted:
                label = 'FAIL_NOHANDOVER'
            elif a_touched:
                label = 'FAIL_TOUCHED_NOLIFT'
            else:
                label = 'FAIL_NOLIFT'
            imageio.mimsave(str(videos_dir / f'episode_{ep_idx:03d}_{label}.mp4'),
                            frames, fps=20)

        results['episodes'].append({
            'episode': ep_idx,
            'control_steps': control_step,
            'success': bool(success),
            'transfer_done': bool(transfer_done),
            'transfer_step': transfer_step,
            'dropped': bool(dropped),
            'settle_run': settle_run,
            'a_lifted': bool(a_lifted),
            'a_touched': bool(a_touched),
            'b_touched': bool(b_touched),
            'final_box_z': float(data.qpos[scene['box_qposadr'] + 2]),
            'max_box_z': max_box_z,
            'min_a_box_dist': (None if min_a_box_dist == float('inf')
                               else min_a_box_dist),
            'max_handover_run': max_handover_run,
            'best_home_err': None if best_home_err == float('inf') else best_home_err,
        })
        results['task_successes'] += int(success)
        results['transfer_count'] += int(transfer_done)
        results['drop_count'] += int(dropped)
        results['a_lifted_count'] += int(a_lifted)
        results['a_touched_count'] += int(a_touched)
        results['b_touched_count'] += int(b_touched)

    renderer.close()

    scored = len(results['episodes'])
    denom = max(scored, 1)
    transferred = [e for e in results['episodes'] if e['transfer_step'] is not None]
    dists = [e['min_a_box_dist'] for e in results['episodes']
             if e['min_a_box_dist'] is not None]

    results['summary'] = {
        'n_episodes': n_episodes,
        'n_scored': scored,
        'success_rate': 100.0 * results['task_successes'] / denom,
        'transfer_rate': 100.0 * results['transfer_count'] / denom,
        'drop_rate': 100.0 * results['drop_count'] / denom,
        'drop_given_transfer': (100.0 * results['drop_count'] / len(transferred)
                                if transferred else None),
        # THE number this script exists to produce.
        'a_lift_rate': 100.0 * results['a_lifted_count'] / denom,
        'a_touch_rate': 100.0 * results['a_touched_count'] / denom,
        'b_touch_rate': 100.0 * results['b_touched_count'] / denom,
        'min_a_box_dist_median': float(np.median(dists)) if dists else None,
        'min_a_box_dist_best': float(np.min(dists)) if dists else None,
        'mean_control_steps': float(np.mean([e['control_steps']
                                             for e in results['episodes']])) if scored else None,
        'mean_transfer_step': (float(np.mean([e['transfer_step'] for e in transferred]))
                               if transferred else None),
    }
    results['octo_diagnostics'] = policy_a.diagnostics()
    if policy_b is not None:
        results['octo_diagnostics_b'] = policy_b.diagnostics()

    s = results['summary']
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f"A LIFTED THE BOX:  {results['a_lifted_count']}/{scored} = "
          f"{s['a_lift_rate']:.1f}%    <-- the pickup rate")
    print(f"  A touched it:    {s['a_touch_rate']:.1f}%")
    if s['min_a_box_dist_median'] is not None:
        print(f"  closest approach: {s['min_a_box_dist_median']*1000:.0f} mm median, "
              f"{s['min_a_box_dist_best']*1000:.0f} mm best")
    print(f"Success:           {s['success_rate']:.1f}%   ({crit_desc})")
    print(f"  Transfer only:   {s['transfer_rate']:.1f}%")
    print(f"  Dropped after:   {s['drop_rate']:.1f}%")
    print(f"  B touched:       {s['b_touch_rate']:.1f}%")
    if s['mean_control_steps'] is not None:
        print(f"Mean control steps: {s['mean_control_steps']:.1f}")
    if results['skipped']:
        print(f"Skipped (grasp lost at reset): {results['skipped']}")

    for tag, key in (('A', 'octo_diagnostics'), ('B', 'octo_diagnostics_b')):
        d = results.get(key)
        if not d:
            continue
        print(f'\nRaw Octo actions, arm {tag} (before scaling):')
        print(f"  position delta norm: median {d['raw_delta_norm_median']:.5f} m, "
              f"p95 {d['raw_delta_norm_p95']:.5f}, max {d['raw_delta_norm_max']:.5f}")
        print(f"  after --action-scale: {d['scaled_delta_mm_median']:.2f} mm "
              f"per control step ({d['scaled_delta_mm_median']*50:.0f} mm/s)")
        print(f"  gripper column: mean {d['gripper_raw_mean']:.3f}, "
              f"{d['gripper_raw_frac_above_half']*100:.0f}% above 0.5, "
              f"state flips on {d['gripper_chunk_flip_rate']*100:.0f}% of chunks")
        # Total commanded PATH LENGTH over one episode, against the distance
        # the gripper actually has to cover. Net displacement is always less
        # than path, so a budget below the reach is a hard proof of failure
        # that has nothing to do with Octo's competence.
        budget = d['scaled_delta_mm_median'] / 1000.0 * max_control_steps
        print(f"  travel budget: {budget*100:.1f} cm of path over "
              f"{max_control_steps} control steps")
        if d.get('box_alignment_mean') is not None:
            print(f"  aim at box: cos {d['box_alignment_mean']:+.3f}, "
                  f"{d['box_alignment_frac_positive']*100:.0f}% of chunks toward it")
            if abs(d['box_alignment_mean']) < 0.15:
                print('  !! Undirected: the commanded motion is uncorrelated '
                      'with where the box is. More travel budget cannot fix '
                      'this -- the visual input is not driving the actions.')
            elif d['box_alignment_mean'] <= -0.15:
                print('  !! Systematically AWAY from the box. Suspect an axis '
                      'or frame convention, not the policy.')
        if budget < 0.42:
            print(f'  !! The gripper starts ~42 cm from the box, so {budget*100:.1f} cm '
                  'of path cannot reach it even if every delta pointed straight '
                  'at it. Raise --action-scale or --max-control-steps; this run '
                  'says nothing about Octo.')
        if d['raw_delta_norm_median'] < 1e-4:
            print('  !! Octo is committing to essentially no motion. The IK '
                  'bridge is not the problem here.')
        if d['gripper_raw_frac_above_half'] in (0.0, 1.0):
            print('  !! The gripper column never crosses 0.5, so the hand never '
                  'changes state. Try --gripper-inverted, then check the source '
                  'dataset convention.')
        elif d['gripper_chunk_flip_rate'] > 0.25:
            print('  !! The gripper is effectively a coin flip between chunks. '
                  'A grasp needs it to STAY closed, so no amount of reaching '
                  'will turn into a lift.')

    print('\nReminder: a low number here is ambiguous between "Octo cannot do '
          'this task on this robot" and "one of the three conversions in the '
          'bridge is wrong". A high number is the unambiguous, useful outcome.')

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out_path}')
    if save_videos:
        print(f'Videos saved to: {videos_dir}/')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--octo-checkpoint', type=str, default='octo-small',
                        help="'octo-small', 'octo-base', an hf:// path, or a "
                             "local finetune directory")
    parser.add_argument('--octo-dataset', type=str, default='bridge_dataset',
                        help='source dataset whose action statistics unnormalise '
                             "Octo's output. Ignored for single-dataset finetunes.")
    parser.add_argument('--action-scale', type=float, default=0.1,
                        help='metres of delta EE per control step per unit of Octo '
                             "output. Absorbs the rate mismatch between Octo's ~5 Hz "
                             'source data and this 50 Hz loop. UNCALIBRATED -- sweep it.')
    parser.add_argument('--instruction', type=str, default=INSTRUCTION_A,
                        help='language instruction given to Octo. This is the only '
                             'task signal it gets, so it matters.')
    parser.add_argument('--camera', type=str, default='wrist_cam_left',
                        help="camera fed to Octo as image_primary. wrist_cam_left is "
                             "what the CoLA features were extracted from, but if the "
                             "wrist cam cannot see the box from the neutral pose the "
                             "test is unfair -- try overhead_cam.")
    parser.add_argument('--wrist-orientation', choices=('grasp', 'initial'),
                        default='grasp',
                        help="orientation the IK pins the wrist to, since Octo's "
                             "rotation deltas are discarded. 'grasp': the "
                             "expert's GRASP_ORIENTATION, which costs a ~78 deg "
                             "uncommanded slew in the first chunk. 'initial': "
                             "hold the pose the episode starts in.")
    parser.add_argument('--random-actions', action='store_true',
                        help='CONTROL CONDITION: draw actions from the source '
                             "dataset's action marginal instead of querying "
                             'Octo. Everything else -- bridge, IK, scoring -- is '
                             'identical, so any gap between this and a real run '
                             "is what Octo's conditioning actually buys.")
    parser.add_argument('--camera-b', type=str, default=None,
                        help="camera fed to arm B's Octo instance. Default "
                             "mirrors --camera: wrist_cam_left -> wrist_cam_right, "
                             "anything else is shared by both arms.")
    parser.add_argument('--instruction-b', type=str, default=INSTRUCTION_B,
                        help="language instruction for arm B. Octo has no partner "
                             "channel, so this is the ONLY way the two arms are "
                             "told they have different jobs.")
    parser.add_argument('--gripper-inverted', action='store_true',
                        help="source dataset uses high=closed rather than high=open")
    parser.add_argument('--arm-b', choices=('hold', 'octo'), default='hold',
                        help="hold: arm B stays put (default; cleanest pickup test, "
                             "B is only a collision hazard). octo: B runs the same "
                             "policy, for the full two-arm picture.")
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--scene-xml', type=str, default=SCENE_XML)
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--handover-only', action='store_true',
                        help='start each episode with arm A already holding the box. '
                             'Note this SKIPS the pickup entirely, so a_lift_rate '
                             'stops being meaningful -- use it only to test the '
                             'transfer half.')
    parser.add_argument('--probe-actions', action='store_true',
                        help='sample raw Octo actions with no IK and no physics, '
                             'report their scale, and exit. Run this FIRST.')
    parser.add_argument('--video-dir', type=str, default='experiments/octo_baseline/videos')
    parser.add_argument('--results-path', type=str,
                        default='experiments/octo_baseline/logs/octo_eval_results.json')
    parser.add_argument('--max-control-steps', type=int, default=350)
    parser.add_argument('--criterion', choices=CRITERIA, default='transfer',
                        help='defaults to transfer here, not hold: a single-policy '
                             'baseline is very unlikely to reach phase 2, and the '
                             'looser criterion keeps the row comparable to the paper.')
    parser.add_argument('--return-arm', choices=('a', 'b'), default='b')
    parser.add_argument('--video-camera', type=str, default=VIDEO_CAMERA)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='shift the per-episode seeds by this amount, as '
                             'cola_eval_aloha.py --seed-offset does. Episodes '
                             'are seeded seed_offset + ep_idx, so disjoint '
                             'offsets give independent evaluation sets of the '
                             'same policy. Distinct from --seed, which only '
                             "seeds the policy's own sampling.")
    args = parser.parse_args()

    checkpoint = OCTO_CHECKPOINTS.get(args.octo_checkpoint, args.octo_checkpoint)

    print('=' * 60)
    print('PRETRAINED OCTO ROLLOUT EVALUATION')
    print('=' * 60)
    print(f'Model: {checkpoint}')
    print(f'Episodes: {args.episodes}')

    print('\n1. Loading Octo...')
    policy = OctoPolicy(
        checkpoint=checkpoint,
        dataset_name=args.octo_dataset,
        scene_xml=args.scene_xml,
        action_scale=args.action_scale,
        instruction=args.instruction,
        camera=args.camera,
        gripper_open_high=not args.gripper_inverted,
        seed=args.seed,
        wrist_orientation=args.wrist_orientation,
        random_actions=args.random_actions,
    )

    if args.probe_actions:
        scene = build_scene(args.scene_xml)
        probe_actions(policy, scene, args.scene_xml,
                      n_episodes=min(args.episodes, 10),
                      handover_only=args.handover_only)
        raise SystemExit(0)

    # Arm B gets its OWN instance. Octo is single-arm -- one camera, one
    # instruction, one 7-d output -- so a shared object would mean a shared
    # observation window and both arms conditioned on arm A's view.
    policy_b = None
    if args.arm_b == 'octo':
        camera_b = args.camera_b
        if camera_b is None:
            camera_b = ('wrist_cam_right' if args.camera == 'wrist_cam_left'
                        else args.camera)
        print('\n1b. Loading Octo for arm B...')
        policy_b = OctoPolicy(
            checkpoint=checkpoint,
            dataset_name=args.octo_dataset,
            scene_xml=args.scene_xml,
            action_scale=args.action_scale,
            instruction=args.instruction_b,
            camera=camera_b,
            gripper_open_high=not args.gripper_inverted,
            wrist_orientation=args.wrist_orientation,
            random_actions=args.random_actions,
            # A different stream, so the two arms do not draw identical
            # samples from identical observations.
            seed=args.seed + 1,
        )

    evaluate(
        policy_a=policy,
        policy_b=policy_b,
        n_episodes=args.episodes,
        scene_xml=args.scene_xml,
        save_videos=not args.no_videos,
        video_dir=args.video_dir,
        results_path=args.results_path,
        max_control_steps=args.max_control_steps,
        video_camera=args.video_camera,
        criterion=args.criterion,
        return_arm=args.return_arm,
        handover_only=args.handover_only,
        arm_b_mode=args.arm_b,
        seed_offset=args.seed_offset,
    )