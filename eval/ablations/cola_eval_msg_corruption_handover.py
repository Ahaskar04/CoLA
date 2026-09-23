"""
Message-corruption ablation for the COLA handover task.

Identical to cola_eval.py (same checkpoint, same env, same episode seeds,
same success criteria) EXCEPT that a fraction of each message's 64
dimensions is zeroed out before the message reaches the decoder.

Normal eval:        full 64-dim messages exchanged between agents.
This ablation:      after the encoders produce msg_a / msg_b, a fresh
                    random mask zeros out --corruption_rate of the dims
                    each step (the rest pass through unchanged).

Different from the random-message baseline: that replaces the *whole*
message with noise. This keeps most of the real message intact and only
drops some dimensions -- a test of robustness to *partial* communication
failure, not a test of whether content matters at all.

The model is NOT retrained -- eval-time message perturbation only.

Usage:
    python3 cola_eval_msg_corruption_handover.py \
        --checkpoint experiments/run_handover_01/checkpoints/best_model.pkl \
        --n_episodes 200 \
        --corruption_rate 0.10 \
        --output_dir eval_msg_corruption/rate_10/
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
import jax.numpy as jnp
from tqdm import tqdm
import imageio

# Make HandoverEnv importable from its package
_HAND_GRIPPER = Path('/home/users/ntu/ahaskar0/cola-research/handover/hand-gripper').resolve()
if str(_HAND_GRIPPER) not in sys.path:
    sys.path.insert(0, str(_HAND_GRIPPER))
from env.handover_env import HandoverEnv, HandoverConfig  # noqa: E402

from cola_architecture import COLAModel, CHUNK_SIZE  # noqa: E402


def corrupt_message(msg, corruption_rate, rng):
    """
    Zero out `corruption_rate` fraction of a message's dimensions.

    A fresh Bernoulli mask is drawn from `rng` each call -- each kept dim
    is multiplied by 1, each dropped dim by 0 (dims are zeroed, NOT
    replaced with noise). `rng` must be a np.random.Generator that is
    completely separate from the environment RNG.
    """
    if corruption_rate <= 0:
        return msg
    mask = rng.binomial(1, 1 - corruption_rate, size=msg.shape).astype(np.float32)
    return msg * mask


def forward_corrupt_message(cola: COLAModel, image_a, image_b,
                            corruption_rate, corruption_rng):
    """
    COLA forward pass with symmetric message passing, but with both
    messages partially corrupted.

    Mirrors COLAModel.forward() exactly, except msg_a / msg_b each have
    `corruption_rate` of their dimensions zeroed (fresh mask per call)
    before being passed to the decoders.
    """
    params = cola.params

    features_a = cola.extract_octo_features(image_a)
    features_b = cola.extract_octo_features(image_b)

    msg_a = cola.encoder_a.apply(params['encoder_a'], features_a)
    msg_b = cola.encoder_b.apply(params['encoder_b'], features_b)

    # --- the only change vs. cola_eval.py: zero out a fraction of dims ---
    # corruption_rng is separate from the env RNG; a fresh mask is drawn
    # every step (masks are NOT reused across timesteps).
    msg_a = jnp.asarray(corrupt_message(np.asarray(msg_a), corruption_rate, corruption_rng))
    msg_b = jnp.asarray(corrupt_message(np.asarray(msg_b), corruption_rate, corruption_rng))
    # --------------------------------------------------------------------

    decoded_b = cola.decoder_a.apply(params['decoder_a'], msg_b)  # A decodes B
    decoded_a = cola.decoder_b.apply(params['decoder_b'], msg_a)  # B decodes A

    combined_a = jnp.concatenate([features_a, decoded_b], axis=-1)
    combined_b = jnp.concatenate([features_b, decoded_a], axis=-1)

    action_a = cola.action_head_a.apply(params['action_head_a'], combined_a)
    action_b = cola.action_head_b.apply(params['action_head_b'], combined_b)

    return action_a, action_b


def evaluate_msg_corruption(
    model_path: str,
    n_episodes: int = 200,
    corruption_rate: float = 0.0,
    corruption_seed: int = 42,
    save_videos: bool = True,
    save_results: bool = True,
    output_dir: str = 'eval_msg_corruption',
    max_steps: int = 600,
) -> Dict:
    print('=' * 60)
    print('COLA HANDOVER EVALUATION -- MESSAGE-CORRUPTION ABLATION')
    print('=' * 60)
    print(f'Model: {model_path}')
    print(f'Episodes: {n_episodes}')
    print(f'Save videos: {save_videos}')
    n_dims_zeroed = int(round(corruption_rate * 64))
    print(f'Corruption rate: {corruption_rate:.0%}  '
          f'(~{n_dims_zeroed}/64 dims zeroed per message per step)')
    print(f'Corruption RNG seed: {corruption_seed} (separate from env RNG)')

    out_dir = Path(output_dir)
    videos_dir = out_dir / 'evaluation_videos'
    results_path = out_dir / 'cola_eval_msg_corruption_results.json'
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

    # Corruption RNG: completely separate from the env RNG. Seeded once per
    # run; a fresh mask is drawn every step (masks NOT reused across steps).
    # env.reset(seed=ep_idx) is untouched -- identical episode seeds to
    # cola_eval.py.
    corruption_rng = np.random.default_rng(corruption_seed)

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
            # Query the model once for the next CHUNK_SIZE actions.
            image_a = obs['image_a'][np.newaxis, ...]
            image_b = obs['image_b'][np.newaxis, ...]

            chunk_a, chunk_b = forward_corrupt_message(
                cola, image_a, image_b, corruption_rate, corruption_rng)
            chunk_a = np.array(chunk_a[0])  # (CHUNK_SIZE, 7)
            chunk_b = np.array(chunk_b[0])  # (CHUNK_SIZE, 7)

            # Execute the whole chunk before re-querying (same as cola_eval.py).
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
        'ablation': 'message_corruption',
        'corruption_rate': corruption_rate,
        'corruption_seed': corruption_seed,
        'dims_zeroed_per_msg': n_dims_zeroed,
        'task_successes': results['task_successes'],
        'a_grasped_count': a_grasped_count,
        'overall_success_rate': overall_success_rate,
        'conditional_success_rate': conditional_success_rate,
        'mean_episode_length': float(np.mean(results['episode_lengths'])),
        'std_episode_length': float(np.std(results['episode_lengths'])),
    }

    print('\n' + '=' * 60)
    print('Message corruption ablation -- handover task')
    print('=' * 60)
    print(f'Corruption rate: {corruption_rate:.0%}')
    print(f'Episodes: {n_episodes}')
    print(f'Conditional success rate: {conditional_success_rate:.1f}%  '
          f'(given A grasped; {results["task_successes"]}/{a_grasped_count})')
    print(f'Overall success rate: {overall_success_rate:.1f}%  '
          f'({results["task_successes"]}/{n_episodes})')
    print(f'Mean episode length: {np.mean(results["episode_lengths"]):.1f} steps')
    for reason, count in results['failure_reasons'].items():
        if count:
            print(f'  failure {reason}: {count} ({100.0 * count / n_episodes:.1f}%)')

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
    parser.add_argument('--corruption_rate', type=float, default=0.0,
                        help='fraction of each message\'s 64 dims to zero out '
                             'per step (0.10 = 10%%, ~6-7 dims; 0.50 = 32 dims)')
    parser.add_argument('--corruption_seed', type=int, default=42,
                        help='seed for the corruption RNG (separate from the '
                             'env RNG; env/episode seeds are unchanged)')
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--output_dir', type=str, default='eval_msg_corruption')
    args = parser.parse_args()

    evaluate_msg_corruption(
        model_path=args.checkpoint,
        n_episodes=args.n_episodes,
        corruption_rate=args.corruption_rate,
        corruption_seed=args.corruption_seed,
        save_videos=not args.no_videos,
        output_dir=args.output_dir,
    )
