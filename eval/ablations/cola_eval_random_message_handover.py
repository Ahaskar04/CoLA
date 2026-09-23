"""
Random-message baseline for the COLA handover task.

Identical to cola_eval.py (same env, same episode seeds, same success
criteria) EXCEPT the learned 64-dim messages are discarded: after the
encoders produce msg_a / msg_b, both are replaced with fresh Gaussian
noise (a new jax.random key every step) before being handed to the
decoders. The rest of the forward pass is untouched.

Purpose: confirm that *message content* drives coordination, not just
the presence of a channel. If random messages still score ~L2 COLA
level, the decoder is ignoring messages and the COLA result is not
meaningful -- this script flags that case.

Usage:
    python3 cola_eval_random_message_handover.py \
        --checkpoint experiments/run_handover_01/checkpoints/best_model.pkl \
        --n_episodes 200 \
        --output_dir eval_random_message/
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
import jax
import jax.numpy as jnp
from tqdm import tqdm
import imageio

# Make HandoverEnv importable from its package
_HAND_GRIPPER = Path('/home/users/ntu/ahaskar0/cola-research/handover/hand-gripper').resolve()
if str(_HAND_GRIPPER) not in sys.path:
    sys.path.insert(0, str(_HAND_GRIPPER))
from env.handover_env import HandoverEnv, HandoverConfig  # noqa: E402

from cola_architecture import COLAModel, CHUNK_SIZE  # noqa: E402


def forward_random_message(cola: COLAModel, image_a, image_b, rng):
    """
    COLA forward pass with symmetric message passing, but with both
    messages replaced by fresh Gaussian noise.

    Mirrors COLAModel.forward() exactly, except msg_a / msg_b are
    overwritten with jax.random.normal samples (matching their shape)
    before being passed to the decoders.

    Returns (action_a, action_b, rng) -- rng is the advanced key so the
    caller gets fresh noise on the next step.
    """
    params = cola.params

    features_a = cola.extract_octo_features(image_a)
    features_b = cola.extract_octo_features(image_b)

    # Messages are still computed normally (to get the correct shape /
    # to mirror the real forward pass), then thrown away.
    msg_a = cola.encoder_a.apply(params['encoder_a'], features_a)
    msg_b = cola.encoder_b.apply(params['encoder_b'], features_b)

    # --- the only change vs. cola_eval.py: replace messages with noise ---
    rng, key_a, key_b = jax.random.split(rng, 3)
    msg_a = jax.random.normal(key_a, shape=msg_a.shape)
    msg_b = jax.random.normal(key_b, shape=msg_b.shape)
    # --------------------------------------------------------------------

    decoded_b = cola.decoder_a.apply(params['decoder_a'], msg_b)  # A decodes B
    decoded_a = cola.decoder_b.apply(params['decoder_b'], msg_a)  # B decodes A

    combined_a = jnp.concatenate([features_a, decoded_b], axis=-1)
    combined_b = jnp.concatenate([features_b, decoded_a], axis=-1)

    action_a = cola.action_head_a.apply(params['action_head_a'], combined_a)
    action_b = cola.action_head_b.apply(params['action_head_b'], combined_b)

    return action_a, action_b, rng


def evaluate_random_message(
    model_path: str,
    n_episodes: int = 200,
    save_videos: bool = True,
    save_results: bool = True,
    output_dir: str = 'eval_random_message',
    max_steps: int = 600,
    seed: int = 0,
) -> Dict:
    print('=' * 60)
    print('COLA HANDOVER EVALUATION -- RANDOM MESSAGE BASELINE')
    print('=' * 60)
    print(f'Model: {model_path}')
    print(f'Episodes: {n_episodes}')
    print(f'Save videos: {save_videos}')
    print('Messages: replaced with fresh Gaussian noise every step')

    out_dir = Path(output_dir)
    videos_dir = out_dir / 'evaluation_videos'
    results_path = out_dir / 'cola_eval_random_message_results.json'
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
    print('\n2. Creating HandoverEnv...')
    env = HandoverEnv(config=HandoverConfig())
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

    # Master RNG for message noise. Episode/env seeds are unchanged
    # (env.reset(seed=ep_idx)) so the only source of variation vs.
    # cola_eval.py is the message noise.
    rng = jax.random.PRNGKey(seed)

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
            # Query the model once for the next CHUNK_SIZE actions
            image_a = obs['image_a'][np.newaxis, ...]
            image_b = obs['image_b'][np.newaxis, ...]

            chunk_a, chunk_b, rng = forward_random_message(cola, image_a, image_b, rng)
            chunk_a = np.array(chunk_a[0])  # (CHUNK_SIZE, 7)
            chunk_b = np.array(chunk_b[0])  # (CHUNK_SIZE, 7)

            # Execute the whole chunk before re-querying.
            for k in range(CHUNK_SIZE):
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
        'baseline': 'random_message',
        'task_successes': results['task_successes'],
        'a_grasped_count': a_grasped_count,
        'overall_success_rate': overall_success_rate,
        'conditional_success_rate': conditional_success_rate,
        'mean_episode_length': float(np.mean(results['episode_lengths'])),
        'std_episode_length': float(np.std(results['episode_lengths'])),
    }

    print('\n' + '=' * 60)
    print('Random message baseline -- handover task')
    print('=' * 60)
    print(f'Episodes: {n_episodes}')
    print(f'Conditional success rate: {conditional_success_rate:.1f}%  '
          f'(given A grasped; {results["task_successes"]}/{a_grasped_count})')
    print(f'Overall success rate: {overall_success_rate:.1f}%  '
          f'({results["task_successes"]}/{n_episodes})')
    print(f'Mean episode length: {np.mean(results["episode_lengths"]):.1f} steps')
    for reason, count in results['failure_reasons'].items():
        if count:
            print(f'  failure {reason}: {count} ({100.0 * count / n_episodes:.1f}%)')

    print('\nCompare to:')
    print('  L0 base Octo:     0%')
    print('  L1 shared obs:    Y%  (pending)')
    print('  L2 COLA (64-dim): 65%')
    print(f'  Random messages:  {overall_success_rate:.1f}%  <- this run')

    if overall_success_rate >= 55.0:
        print('\n[WARNING] Random messages perform at ~L2 COLA level. This '
              'suggests the decoder is ignoring message content entirely '
              '-- the COLA result may not be meaningful. Investigate before '
              'trusting the 65% number.')

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
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--output_dir', type=str, default='eval_random_message')
    parser.add_argument('--seed', type=int, default=0,
                        help='master RNG seed for message noise (env seeds unchanged)')
    args = parser.parse_args()

    evaluate_random_message(
        model_path=args.checkpoint,
        n_episodes=args.n_episodes,
        save_videos=not args.no_videos,
        output_dir=args.output_dir,
        seed=args.seed,
    )
