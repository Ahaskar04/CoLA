"""
No-action-chunking ablation for the COLA handover task.

Identical to cola_eval.py (same checkpoint, same env, same episode seeds,
same success criteria) EXCEPT for how many actions are executed per
forward pass.

Normal eval: the model predicts 10 actions, all 10 are executed before
re-querying the backbone / exchanging messages again (chunk N=10).

This ablation: --chunk_size controls how many of the predicted actions
are actually executed before re-querying. With --chunk_size 1 the model
still outputs 10 actions, but actions 2-10 are discarded and only action
1 is executed -- so the backbone runs and messages are exchanged every
single step.

The model is NOT retrained. It was trained with N=10; this only changes
eval-time execution.

Purpose: quantify how much action chunking contributes to coordination.

Usage:
    python3 cola_eval_no_chunking_handover.py \
        --checkpoint experiments/run_handover_01/checkpoints/best_model.pkl \
        --n_episodes 200 \
        --chunk_size 1 \
        --output_dir eval_no_chunking/
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm
import imageio

# Make HandoverEnv importable from its package
_HAND_GRIPPER = Path('/home/users/ntu/ahaskar0/cola-research/handover/hand-gripper').resolve()
if str(_HAND_GRIPPER) not in sys.path:
    sys.path.insert(0, str(_HAND_GRIPPER))
from env.handover_env import HandoverEnv, HandoverConfig  # noqa: E402

# CHUNK_SIZE here is the model's *output* chunk (always 10 -- that's what
# the checkpoint was trained with). The number of actions actually
# executed per forward pass is a separate eval-time knob (--chunk_size).
from cola_architecture import COLAModel, CHUNK_SIZE as MODEL_CHUNK_SIZE  # noqa: E402


def evaluate_no_chunking(
    model_path: str,
    n_episodes: int = 200,
    exec_chunk_size: int = 1,
    save_videos: bool = True,
    save_results: bool = True,
    output_dir: str = 'eval_no_chunking',
    max_steps: int = 600,
    env_max_steps: int = 500,
) -> Dict:
    print('=' * 60)
    print('COLA HANDOVER EVALUATION -- NO-CHUNKING ABLATION')
    print('=' * 60)
    print(f'Model: {model_path}')
    print(f'Episodes: {n_episodes}')
    print(f'Save videos: {save_videos}')
    print(f'Model output chunk: {MODEL_CHUNK_SIZE} actions/forward pass (trained)')
    print(f'Executed per forward pass: {exec_chunk_size} '
          f'(actions {exec_chunk_size + 1}-{MODEL_CHUNK_SIZE} discarded)')
    print(f'Env episode cap (HandoverConfig.max_steps): {env_max_steps} steps')
    if env_max_steps != 500:
        print(f'  NOTE: default is 500; this run uses {env_max_steps}. Success '
              f'rate is NOT directly comparable to the 500-step 65% baseline.')
    if exec_chunk_size > MODEL_CHUNK_SIZE:
        raise ValueError(
            f'--chunk_size {exec_chunk_size} exceeds model output chunk '
            f'{MODEL_CHUNK_SIZE}; cannot execute more actions than predicted.'
        )

    out_dir = Path(output_dir)
    videos_dir = out_dir / 'evaluation_videos'
    results_path = out_dir / 'cola_eval_no_chunking_results.json'
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained COLA model (same checkpoint as normal eval)
    print('\n1. Loading COLA model...')
    cola = COLAModel()
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    cola.params = ckpt['params']
    print('   model loaded')

    # 2. Create environment
    # env_max_steps overrides HandoverConfig.max_steps (default 500), which
    # is the env's *internal* truncation cap -- the real limit on episode
    # length. The script-level `max_steps` below is only a secondary outer
    # guard and never fires unless it is set lower than env_max_steps.
    print('\n2. Creating HandoverEnv...')
    env = HandoverEnv(config=HandoverConfig(max_steps=env_max_steps))
    print('   env ready')

    # 3. Run episodes
    print(f'\n3. Running {n_episodes} episodes...')
    results = {
        'episodes': [],
        'task_successes': 0,
        'episode_lengths': [],
        'failure_reasons': {
            'object_fell': 0,
            'timeout': 0,
            'goal_reached_without_handover': 0,
            'other': 0,
        },
    }

    # Handover validation: require B to actually hold the object at some point
    # (matches collect_demos.py / cola_eval.py logic)
    GRIPPER_CLOSE_THRESHOLD = 0.048  # gripper closed (holding) when state < this
    OBJECT_NEAR_GRIPPER_THRESHOLD = 0.06  # 6cm

    # how many episodes A grasped the object (B held it) -- denominator
    # for the conditional success rate
    a_grasped_count = 0

    for ep_idx in tqdm(range(n_episodes), desc='Evaluating'):
        obs = env.reset(seed=ep_idx)
        step = 0
        frames = []
        terminated = False
        truncated = False
        info = {}
        b_held_object = False

        while not (terminated or truncated) and step < max_steps:
            # Query the model. It always predicts MODEL_CHUNK_SIZE (=10)
            # actions; we only execute the first `exec_chunk_size` of them
            # before re-querying (backbone + message exchange).
            image_a = obs['image_a'][np.newaxis, ...]
            image_b = obs['image_b'][np.newaxis, ...]

            chunk_a, chunk_b = cola.forward(image_a, image_b)
            chunk_a = np.array(chunk_a[0])  # (MODEL_CHUNK_SIZE, 7)
            chunk_b = np.array(chunk_b[0])  # (MODEL_CHUNK_SIZE, 7)

            # Execute only the first `exec_chunk_size` actions, then break
            # out to re-query. Also break early on success/timeout/failure.
            for k in range(exec_chunk_size):
                if step >= max_steps:
                    break
                obs, _reward, terminated, truncated, info = env.step(chunk_a[k], chunk_b[k])
                step += 1

                if save_videos:
                    frames.append(env.render_overhead())

                if not b_held_object:
                    gripper_b_pos = info.get('gripper_b_position', np.zeros(3))
                    obj_pos_now = info.get('object_position', np.zeros(3))
                    dist_b_to_obj = float(np.linalg.norm(gripper_b_pos - obj_pos_now))
                    gripper_b_state = float(obs['state_b'][6])
                    if (dist_b_to_obj < OBJECT_NEAR_GRIPPER_THRESHOLD
                            and gripper_b_state < GRIPPER_CLOSE_THRESHOLD):
                        b_held_object = True

                if terminated:
                    break
                if truncated:
                    break

        # Real success: goal zone reached AND B actually handled the object
        success = bool(terminated and b_held_object)

        if b_held_object:
            a_grasped_count += 1

        # Save video
        if save_videos and frames:
            if success:
                label = 'SUCCESS'
            elif terminated and not b_held_object:
                label = 'FAIL_NOHANDOVER'
            else:
                label = 'FAIL'
            out = videos_dir / f'episode_{ep_idx:03d}_{label}.mp4'
            imageio.mimsave(str(out), frames, fps=25)

        # Record per-episode info
        ep_record = {
            'episode': ep_idx,
            'steps': step,
            'success': bool(success),
            'terminated': bool(terminated),
            'b_held_object': bool(b_held_object),
            'final_object_y': float(info.get('object_position', [0, 0, 0])[1]) if info else 0.0,
            'object_past_goal_line': bool(info.get('object_past_goal_line', False)) if info else False,
        }
        results['episodes'].append(ep_record)
        results['episode_lengths'].append(step)

        if success:
            results['task_successes'] += 1
        else:
            if terminated and not b_held_object:
                results['failure_reasons']['goal_reached_without_handover'] += 1
            elif info.get('failure', False):
                results['failure_reasons']['object_fell'] += 1
            elif truncated or step >= max_steps:
                results['failure_reasons']['timeout'] += 1
            else:
                results['failure_reasons']['other'] += 1

    env.close()

    # 4. Summary
    overall_success_rate = 100.0 * results['task_successes'] / n_episodes
    conditional_success_rate = (
        100.0 * results['task_successes'] / a_grasped_count
        if a_grasped_count > 0 else 0.0
    )
    results['summary'] = {
        'n_episodes': n_episodes,
        'ablation': 'no_action_chunking',
        'model_output_chunk_size': MODEL_CHUNK_SIZE,
        'exec_chunk_size': exec_chunk_size,
        'task_successes': results['task_successes'],
        'a_grasped_count': a_grasped_count,
        'overall_success_rate': overall_success_rate,
        'conditional_success_rate': conditional_success_rate,
        'mean_episode_length': float(np.mean(results['episode_lengths'])),
        'std_episode_length': float(np.std(results['episode_lengths'])),
    }

    print('\n' + '=' * 60)
    print('No action chunking ablation -- handover task')
    print('=' * 60)
    print(f'Episodes: {n_episodes}')
    print(f'Executed per forward pass: {exec_chunk_size} '
          f'(model still predicts {MODEL_CHUNK_SIZE})')
    print(f'Conditional success rate: {conditional_success_rate:.1f}%  '
          f'(given A grasped; {results["task_successes"]}/{a_grasped_count})')
    print(f'Overall success rate: {overall_success_rate:.1f}%  '
          f'({results["task_successes"]}/{n_episodes})')
    print(f'Mean episode length: {np.mean(results["episode_lengths"]):.1f} steps')
    for reason, count in results['failure_reasons'].items():
        if count:
            print(f'  failure {reason}: {count} ({100.0 * count / n_episodes:.1f}%)')

    print('\nCompare to:')
    print('  Full COLA (chunk N=10):  65%')
    print(f'  No chunking (N={exec_chunk_size}):       '
          f'{overall_success_rate:.1f}%  <- this run')

    if save_results:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {results_path}')

    if save_videos:
        print(f'Videos saved to: {videos_dir}/')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/run_handover_01/checkpoints/best_model.pkl')
    parser.add_argument('--n_episodes', type=int, default=200)
    parser.add_argument('--chunk_size', type=int, default=1,
                        help='actions executed per forward pass (model still '
                             'predicts 10); 1 = no chunking, re-query every step')
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--output_dir', type=str, default='eval_no_chunking')
    parser.add_argument('--env_max_steps', type=int, default=500,
                        help='HandoverConfig.max_steps -- the env\'s internal '
                             'episode truncation cap, and the real limit on '
                             'episode/video length. Default 500 (~20s video '
                             '@ 25fps); 1000 ~= 40s. Changing this makes the '
                             'success rate NOT comparable to the 500-step '
                             '65% baseline.')
    args = parser.parse_args()

    # Script-level guard derived from env cap (+ one chunk of slack) so it
    # never truncates an episode before the env itself does.
    script_max_steps = args.env_max_steps + MODEL_CHUNK_SIZE

    evaluate_no_chunking(
        model_path=args.checkpoint,
        n_episodes=args.n_episodes,
        exec_chunk_size=args.chunk_size,
        save_videos=not args.no_videos,
        output_dir=args.output_dir,
        max_steps=script_max_steps,
        env_max_steps=args.env_max_steps,
    )
