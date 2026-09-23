"""Roll out two DECENTRALISED finetuned Octo policies on the ALOHA handover.

One policy per arm, each finetuned by finetune_octo_arm.py. Each sees ONLY its
own wrist camera and its own joint state, and there is no channel between them:
the only thing telling them apart is the language instruction each was trained
with. That is the comparison CoLA is about.

Why this is a separate script from eval_octo.py: a finetuned policy emits
ABSOLUTE JOINT POSITIONS, exactly the 7-d vector collect_demos.py recorded as
action_a. So the entire delta-EE bridge in eval_octo.py -- mink IK, cumulative
delta integration, GRASP_ORIENTATION pinning, base-frame rotation, action_scale
-- is not merely retuned but deleted. Actions go straight to data.ctrl, which is
where the scripted expert put them. Scoring, scene, seeds and criteria are
imported from eval_octo so the rows stay comparable.

Proprio is built to match state_a in scripted_policy.py exactly:
    concat(qpos[6 arm joints], [qpos[left_finger]])
and normalized with the checkpoint's own dataset_statistics.

    python eval_octo_finetuned.py \
        --checkpoint-a /scratch/.../octo_finetune/arm_a \
        --checkpoint-b /scratch/.../octo_finetune/arm_b \
        --episodes 50 --criterion hold
"""

import argparse
import json
from pathlib import Path

import numpy as np

import eval_octo as E


class FinetunedOctoPolicy:
    """One finetuned Octo driving one arm, in absolute joint space.

    Exposes the same surface as eval_octo.OctoPolicy (reset/act/diagnostics plus
    camera/instruction/action_scale/stats_name) so eval_octo.evaluate can drive
    it unchanged.
    """

    def __init__(self, checkpoint_dir, camera=None, instruction=None, step=None,
                 seed=0, gripper_snap=False):
        import jax
        from octo.model.octo_model import OctoModel

        self.jax = jax
        ckpt = Path(checkpoint_dir)
        meta = {}
        if (ckpt / 'finetune_meta.json').exists():
            meta = json.load(open(ckpt / 'finetune_meta.json'))

        # A run split across queue jobs leaves resume_state.npz until its last
        # segment finishes; evaluating before then scores a half-trained model.
        if (ckpt / 'resume_state.npz').exists():
            raise SystemExit(f'{ckpt} is mid-training (resume_state.npz present)')
        print(f'   loading finetuned {ckpt}' + (f' @ step {step}' if step else ''))
        self.model = OctoModel.load_pretrained(str(ckpt), step=step)

        # Single-dataset finetune, so statistics are flat rather than keyed by
        # source dataset. These are OUR ALOHA statistics, not Bridge's.
        stats = self.model.dataset_statistics
        assert 'action' in stats, (
            f'{ckpt} has per-dataset statistics {sorted(stats)}; this script '
            'expects a single-dataset finetune')
        self.action_stats = stats['action']
        self.p_mean = np.asarray(stats['proprio']['mean'], dtype=np.float32) \
            if 'proprio' in stats else None
        self.p_std = np.asarray(stats['proprio']['std'], dtype=np.float32) \
            if 'proprio' in stats else None

        self.instruction = instruction or meta.get('instruction', '')
        self.task = self.model.create_tasks(texts=[self.instruction])
        # Render the SAME view the policy trained on. Taking this from the
        # checkpoint's own metadata makes a train/eval camera mismatch
        # impossible to introduce by forgetting a flag.
        self.meta = meta
        self.camera = camera or meta.get('camera')
        # Second camera, when the checkpoint was trained with two. Taken from
        # the checkpoint so train and eval cannot disagree about the view.
        self.wrist_camera = meta.get('wrist_camera')
        assert self.camera, (
            f'{ckpt} has no camera in finetune_meta.json; pass --camera-a/-b')
        self.use_proprio = bool(meta.get('use_proprio', True))
        self.window = int(meta.get('window_size', 1))
        self.horizon = int(meta.get('action_horizon', 50))
        # A head shorter than CHUNK_SIZE (octo's native 4) returns a shorter
        # chunk; eval_octo's loop executes only what was predicted and
        # re-queries, so the policy replans more often than CoLA's 10.
        self.exec_steps = min(self.horizon, E.CHUNK_SIZE)
        self.rng = jax.random.PRNGKey(seed)

        # Present for interface compatibility only; there is no scale in an
        # absolute joint action space.
        self.action_scale = 1.0
        self.stats_name = 'finetune'
        # The expert only ever commanded two gripper values (0.002 / 0.037).
        # Anything between is a command the robot never saw in training and
        # leaves the hand half-closed -- able neither to envelop the box nor
        # to grip it. Snapping to the nearer of the two is faithful to the
        # demonstrated action space. Off by default so the raw policy output
        # stays measurable.
        self.gripper_snap = gripper_snap

        self.grip_values = []
        self.grip_chunk_values = []
        self.joint_step = []

        print(f'   wrist camera: {self.wrist_camera or "(none)"}')
        print(f'   arm {meta.get("arm","?")} | camera {camera} | '
              f'window {self.window} | horizon {self.horizon} '
              f'(executing {self.exec_steps}'
              f'{" -- SHORTER than CoLA cadence " + str(E.CHUNK_SIZE) if self.exec_steps < E.CHUNK_SIZE else ""})')
        print(f'   instruction: {self.instruction!r}')

    def reset(self):
        pass                      # window 1: no history to carry across episodes

    def _proprio(self, scene, arm):
        """Exactly state_a/state_b as scripted_policy.py recorded them."""
        d = scene['data']
        return np.concatenate([d.qpos[arm['qadr']], [d.qpos[arm['finger_qadr']]]])

    def act(self, scene, renderer, arm):
        renderer.update_scene(scene['data'], camera=self.camera)
        image = renderer.render()

        obs = {
            'image_primary': image[None, None, ...].astype(np.uint8),
            'timestep_pad_mask': np.ones((1, self.window), dtype=bool),
        }
        if self.wrist_camera:
            renderer.update_scene(scene['data'], camera=self.wrist_camera)
            obs['image_wrist'] = renderer.render()[None, None, ...].astype(np.uint8)
        if self.use_proprio:
            proprio = ((self._proprio(scene, arm) - self.p_mean)
                       / (self.p_std + 1e-8))
            obs['proprio'] = proprio[None, None, ...].astype(np.float32)

        self.rng, sub = self.jax.random.split(self.rng)
        actions = self.model.sample_actions(
            obs, self.task, unnormalization_statistics=self.action_stats, rng=sub)
        actions = np.asarray(actions)[0]              # (horizon, 7)

        # Receding horizon: predict the full chunk, execute only the first
        # CHUNK_SIZE, then re-observe. Matches octo's RHCWrapper and keeps the
        # control cadence identical to CoLA's.
        # np.array (not asarray): asarray on a JAX array returns a
        # READ-ONLY view, and --gripper-snap writes into this.
        chunk = np.array(actions[:E.CHUNK_SIZE], dtype=np.float32)

        # Every EXECUTED step, not just the chunk's last action. Sampling
        # only chunk[-1] mixes two different populations and made a bimodal
        # gripper look like a flat one.
        # Record the RAW prediction before any snapping, so the diagnostic
        # still reports what the policy actually produced.
        self.grip_values.extend(chunk[:, 6].tolist())
        if self.gripper_snap:
            mid = (E.GRIPPER_OPEN + E.GRIPPER_CLOSED) / 2
            chunk[:, 6] = np.where(chunk[:, 6] > mid,
                                   E.GRIPPER_OPEN, E.GRIPPER_CLOSED)
        self.grip_chunk_values.append(float(np.median(chunk[:, 6])))
        self.joint_step.append(
            float(np.abs(np.diff(chunk[:, :6], axis=0)).mean()) if len(chunk) > 1
            else 0.0)
        return chunk

    def diagnostics(self):
        # Deliberately empty: eval_octo's printout is written for the delta-EE
        # bridge and its keys are meaningless here. Real numbers come from
        # joint_diagnostics() below.
        return {}

    def joint_diagnostics(self):
        if not self.grip_values:
            return {}
        g = np.asarray(self.grip_values)
        mid = (E.GRIPPER_OPEN + E.GRIPPER_CLOSED) / 2
        gc = np.asarray(self.grip_chunk_values) > mid
        # Percentiles expose the shape: a bimodal command sits near 0.002 and
        # 0.037 with little between, a hedging one clusters at the mean.
        return {
            'gripper_cmd_mean': float(g.mean()),
            'gripper_frac_open': float((g > mid).mean()),
            'gripper_p10': float(np.percentile(g, 10)),
            'gripper_p50': float(np.percentile(g, 50)),
            'gripper_p90': float(np.percentile(g, 90)),
            'gripper_frac_midband': float(
                ((g > E.GRIPPER_CLOSED + 0.008) & (g < E.GRIPPER_OPEN - 0.008)).mean()),
            'gripper_chunk_flip_rate': (
                float((gc[1:] != gc[:-1]).mean()) if len(gc) > 1 else 0.0),
            'mean_joint_step_rad': float(np.mean(self.joint_step)),
        }


class CentralisedOctoPolicy(FinetunedOctoPolicy):
    """One policy emitting a 14-d bimanual action for BOTH arms.

    This is octo's own ALOHA recipe (examples/02_finetune_new_observation_action.py):
    a single third-person camera, proprio over both arms, and one L1ActionHead
    with action_dim=14. It is the CENTRALISED upper bound -- one policy sees
    everything and commands everything, with no decentralisation and therefore
    no coordination problem to solve.

    eval_octo.evaluate drives two arms by calling act() once per arm per chunk,
    always arm A first. So the 14-d chunk is computed on the A call and the B
    call is served from that same forward pass; querying twice would consume two
    observations for one decision and double the diagnostics.

    Column order follows ARM_SPEC['both'] -- (action_a, action_b) -- so 0:7 is
    arm A and 7:14 is arm B. That ordering must not drift from the data module.
    """

    def _proprio_both(self, scene):
        d = scene['data']
        return np.concatenate([
            np.concatenate([d.qpos[scene[k]['qadr']],
                            [d.qpos[scene[k]['finger_qadr']]]])
            for k in ('a', 'b')])

    def act(self, scene, renderer, arm):
        if arm['prefix'] == 'left':
            renderer.update_scene(scene['data'], camera=self.camera)
            image = renderer.render()
            obs = {
                'image_primary': image[None, None, ...].astype(np.uint8),
                'timestep_pad_mask': np.ones((1, self.window), dtype=bool),
            }
            if self.wrist_camera:
                renderer.update_scene(scene['data'], camera=self.wrist_camera)
                obs['image_wrist'] = (renderer.render()[None, None, ...]
                                      .astype(np.uint8))
            if self.use_proprio:
                pr = (self._proprio_both(scene) - self.p_mean) / (self.p_std + 1e-8)
                obs['proprio'] = pr[None, None, ...].astype(np.float32)

            self.rng, sub_ = self.jax.random.split(self.rng)
            actions = self.model.sample_actions(
                obs, self.task, unnormalization_statistics=self.action_stats,
                rng=sub_)
            chunk = np.array(np.asarray(actions)[0][:E.CHUNK_SIZE],
                             dtype=np.float32)      # (CHUNK_SIZE, 14)

            self.grip_values.extend(chunk[:, 6].tolist())
            if self.gripper_snap:
                mid = (E.GRIPPER_OPEN + E.GRIPPER_CLOSED) / 2
                for col in (6, 13):
                    chunk[:, col] = np.where(chunk[:, col] > mid,
                                             E.GRIPPER_OPEN, E.GRIPPER_CLOSED)
            self.grip_chunk_values.append(float(np.median(chunk[:, 6])))
            self.joint_step.append(
                float(np.abs(np.diff(chunk[:, :6], axis=0)).mean()))
            self._chunk14 = chunk

        lo = 0 if arm['prefix'] == 'left' else 7
        return self._chunk14[:, lo:lo + 7]


# Which wrist FOV each demo cache was rendered at, measured by rendering a
# reconstructed frame at both candidates and comparing to the stored image
# (check_wrist_fov.py). The overhead camera is 1.93e-3 in every scene variant,
# so only wrist-camera checkpoints care -- but for those a mismatch trains on
# one lens and evaluates on another, which looks like a failed ablation rather
# than a bug.
CACHE_WRIST_FOV = {
    'cache_aloha_handover_v2': 58.01,
    'cache_fulltask_v5': 83.44,
    'cache_handover_only_v1': 83.44,
}


def check_wrist_fov(policy, scene_xml):
    """Fail loudly when a wrist-trained checkpoint meets the wrong lens."""
    import numpy as np, mujoco
    cams = [c for c in (getattr(policy, 'camera', None),
                        getattr(policy, 'wrist_camera', None)) if c]
    if not any('wrist' in c for c in cams):
        return
    manifest = (policy.meta or {}).get('manifest', '')
    cache = manifest.rstrip('/').split('/')[-2] if manifest else ''
    want = CACHE_WRIST_FOV.get(cache)
    if want is None:
        print(f'   NOTE: no recorded wrist FOV for {cache!r}; not checked')
        return
    m = mujoco.MjModel.from_xml_path(scene_xml)
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_cam_left')
    ss, fl = m.cam_sensorsize[i], m.cam_intrinsic[i][:2]
    got = float(np.degrees(2 * np.arctan(ss[1] / (2 * fl[1]))))
    if abs(got - want) > 0.5:
        raise SystemExit(
            f'wrist FOV mismatch: {cache} was rendered at {want:.2f} deg but '
            f'{scene_xml} gives {got:.2f} deg. Pass the matching --scene-xml.')
    print(f'   wrist FOV OK: {cache} {want:.2f} deg matches the scene')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-a', required=True)
    p.add_argument('--checkpoint-b', default=None,
                   help='omit together with --arm-b hold for a single-arm '
                        'pickup test')
    p.add_argument('--arm-b', choices=('hold', 'octo'), default='octo',
                   help='hold keeps arm B still, which is the cleanest way to '
                        "measure arm A's pickup rate on its own")
    p.add_argument('--step-a', type=int, default=None)
    p.add_argument('--step-b', type=int, default=None)
    p.add_argument('--camera-a', default=None,
                   help='defaults to the camera recorded in the checkpoint')
    p.add_argument('--camera-b', default=None)
    p.add_argument('--instruction-a', default=None)
    p.add_argument('--instruction-b', default=None)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--criterion', choices=E.CRITERIA, default='hold',
                   help="defaults to hold, matching cola_eval_aloha.py")
    p.add_argument('--handover-only', action='store_true')
    p.add_argument('--scene-xml', default=E.SCENE_XML)
    p.add_argument('--max-control-steps', type=int, default=350)
    p.add_argument('--no-videos', action='store_true')
    p.add_argument('--video-dir', default='experiments/octo_baseline/videos/ft')
    p.add_argument('--results-path',
                   default='experiments/octo_baseline/logs/octo_ft_decentralised.json')
    p.add_argument('--centralised', action='store_true',
                   help='--checkpoint-a is a single 14-d bimanual policy that '
                        'drives BOTH arms (octo ALOHA reference recipe)')
    p.add_argument('--gripper-snap', action='store_true',
                   help='snap the gripper command to the nearer of the two '
                        'values the expert actually used (0.002 / 0.037)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seed-offset', type=int, default=0,
                   help='shift the per-episode seeds by this amount, exactly '
                        'as cola_eval_aloha.py --seed-offset does. Episodes '
                        'are seeded seed_offset + ep_idx, so 0/1000/2000/... '
                        'are independent 200-episode draws of the same policy '
                        'and their spread is the run-to-run variance. Distinct '
                        "from --seed, which seeds the policy's sampling only.")
    args = p.parse_args()

    print('=' * 62)
    print('DECENTRALISED FINETUNED OCTO  (two policies, no comms channel)')
    print('=' * 62)

    print('\n1. Loading policies...')
    if args.centralised:
        pa = CentralisedOctoPolicy(args.checkpoint_a, args.camera_a,
                                   args.instruction_a, args.step_a, args.seed,
                                   args.gripper_snap)
        assert pa.action_stats['mean'].shape[-1] == 14, (
            '--centralised needs a 14-d checkpoint; this one emits '
            f"{pa.action_stats['mean'].shape[-1]}")
        print('   centralised: one 14-d policy drives both arms')
        check_wrist_fov(pa, args.scene_xml)
        results = E.evaluate(
            policy_a=pa, policy_b=pa,          # same object, one forward/chunk
            n_episodes=args.episodes, scene_xml=args.scene_xml,
            save_videos=not args.no_videos, video_dir=args.video_dir,
            results_path=args.results_path,
            max_control_steps=args.max_control_steps,
            criterion=args.criterion, handover_only=args.handover_only,
            arm_b_mode='octo', seed_offset=args.seed_offset)
        d = pa.joint_diagnostics()
        if d:
            print(f'\ncentralised diagnostics: gripper mean '
                  f'{d["gripper_cmd_mean"]:.4f}, {d["gripper_frac_open"]*100:.0f}% '
                  f'open, joint step {d["mean_joint_step_rad"]*1000:.2f} mrad')
        results['finetuned_diagnostics'] = {'centralised': d}
        results['checkpoints'] = {'centralised': args.checkpoint_a}
        with open(args.results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSUCCESS RATE: {results['summary']['success_rate']:.1f}%  |  "
              f"transfer {results['summary']['transfer_rate']:.1f}%  |  "
              f"a_lift {results['summary']['a_lift_rate']:.1f}%")
        print(f'Results -> {args.results_path}')
        return

    pa = FinetunedOctoPolicy(args.checkpoint_a, args.camera_a,
                             args.instruction_a, args.step_a, args.seed,
                             args.gripper_snap)
    check_wrist_fov(pa, args.scene_xml)
    pb = None
    if args.arm_b == 'octo':
        assert args.checkpoint_b, '--arm-b octo needs --checkpoint-b'
        pb = FinetunedOctoPolicy(args.checkpoint_b, args.camera_b,
                                 args.instruction_b, args.step_b, args.seed + 1,
                                 args.gripper_snap)
        check_wrist_fov(pb, args.scene_xml)
    else:
        print('   arm B: held still (single-arm pickup test)')
    print(f'   gripper snap: {args.gripper_snap}')

    results = E.evaluate(
        policy_a=pa, policy_b=pb,
        n_episodes=args.episodes,
        scene_xml=args.scene_xml,
        save_videos=not args.no_videos,
        video_dir=args.video_dir,
        results_path=args.results_path,
        max_control_steps=args.max_control_steps,
        criterion=args.criterion,
        handover_only=args.handover_only,
        arm_b_mode=args.arm_b,
        seed_offset=args.seed_offset,
    )

    da = pa.joint_diagnostics()
    db = pb.joint_diagnostics() if pb is not None else {}
    print('\nFinetuned-policy diagnostics (absolute joint space):')
    for tag, d in (('A', da), ('B', db)):
        if not d:
            continue
        print(f'  arm {tag}: gripper mean {d["gripper_cmd_mean"]:.4f}, '
              f'{d["gripper_frac_open"]*100:.0f}% of executed steps open '
              f'(p10 {d["gripper_p10"]:.4f} p50 {d["gripper_p50"]:.4f} '
              f'p90 {d["gripper_p90"]:.4f}), '
              f'{d["gripper_frac_midband"]*100:.0f}% stuck mid-range, '
              f'flips on {d["gripper_chunk_flip_rate"]*100:.0f}% of chunks, '
              f'joint step {d["mean_joint_step_rad"]*1000:.2f} mrad')

    results['finetuned_diagnostics'] = {'a': da, 'b': db}
    results['checkpoints'] = {'a': args.checkpoint_a, 'b': args.checkpoint_b}
    results['arm_b_mode'] = args.arm_b
    print(f"\nPICKUP RATE (a_lift_rate): {results['summary']['a_lift_rate']:.1f}%")
    with open(args.results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults (with diagnostics) -> {args.results_path}')


if __name__ == '__main__':
    main()
