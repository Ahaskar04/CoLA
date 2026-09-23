"""
Gaussian observation-noise ablation for the COLA handover task.

Identical to cola_eval.py (same checkpoint, same env, same episode seeds,
same success criteria) EXCEPT that Gaussian pixel noise is added to both
camera images before the Octo forward pass.

Normal eval:  clean images -> Octo -> features -> encoder -> message
This ablation: image + N(0, sigma) -> clip to uint8 -> Octo -> ...

Fresh noise is drawn every step (the noise RNG is seeded once per run;
env/episode seeds are unchanged). sigma=0 is a sanity check that should
reproduce the clean ~65% number.

The model is NOT retrained -- eval-time input perturbation only.

Purpose: test whether COLA degrades gracefully under noisy visual input.

Usage:
    python3 cola_eval_obs_noise_handover.py \
        --checkpoint experiments/run_handover_01/checkpoints/best_model.pkl \
        --n_episodes 200 \
        --noise_sigma 10 \
        --output_dir eval_obs_noise/sigma_10/
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

from cola_architecture import COLAModel, CHUNK_SIZE  # noqa: E402


def add_gaussian_noise(image, sigma, rng):
    """Add Gaussian noise to a uint8 image, clip to valid uint8 range."""
    if sigma <= 0:
        return image
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def evaluate_obs_noise(
    model_path: str,
    n_episodes: int = 200,
    noise_sigma: float = 0.0,
    noise_seed: int = 0,
    save_videos: bool = True,
    save_results: bool = True,
    output_dir: str = 'eval_obs_noise',
    max_steps: int = 600,
) -> Dict:
    print('=' * 60)
    print('COLA HANDOVER EVALUATION -- GAUSSIAN OBS-NOISE ABLATION')
    print('=' * 60)
    print(f'Model: {model_path}')
    print(f'Episodes: {n_episodes}')
    print(f'Noise sigma: {noise_sigma}')
    print(f'Save videos: {save_videos}')
    print('Noise: fresh Gaussian pixel noise added to both images every step, '
          'before Octo')

    out_dir = Path(output_dir)
    videos_dir = out_dir / 'evaluation_videos'
    results_path = out_dir / 'cola_eval_obs_noise_results.json'
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

    # Noise RNG: seeded once per run so noise is reproducible, but a fresh
    # draw is taken every step (noise is NOT repeated across steps). Env /
    # episode seeds are untouched -- env.reset(seed=ep_idx) as in cola_eval.py.
    noise_rng = np.random.default_rng(noise_seed)

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
            # Add fresh Gaussian noise to the raw uint8 images BEFORE Octo.
            image_a = add_gaussian_noise(obs['image_a'], noise_sigma, noise_rng)
            image_b = add_gaussian_noise(obs['image_b'], noise_sigma, noise_rng)
            image_a = image_a[np.newaxis, ...]
            image_b = image_b[np.newaxis, ...]

            chunk_a, chunk_b = cola.forward(image_a, image_b)
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
        'ablation': 'gaussian_obs_noise',
        'noise_sigma': noise_sigma,
        'task_successes': results['task_successes'],
        'a_grasped_count': a_grasped_count,
        'overall_success_rate': overall_success_rate,
        'conditional_success_rate': conditional_success_rate,
        'mean_episode_length': float(np.mean(results['episode_lengths'])),
        'std_episode_length': float(np.std(results['episode_lengths'])),
    }

    print('\n' + '=' * 60)
    print('Obs noise ablation -- handover task')
    print('=' * 60)
    print(f'Sigma: {noise_sigma}')
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
    parser.add_argument('--noise_sigma', type=float, default=0.0,
                        help='std dev of Gaussian pixel noise added to images '
                             'before Octo (0 = clean sanity check)')
    parser.add_argument('--noise_seed', type=int, default=0,
                        help='seed for the noise RNG (env/episode seeds unchanged)')
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--output_dir', type=str, default='eval_obs_noise')
    args = parser.parse_args()

    evaluate_obs_noise(
        model_path=args.checkpoint,
        n_episodes=args.n_episodes,
        noise_sigma=args.noise_sigma,
        noise_seed=args.noise_seed,
        save_videos=not args.no_videos,
        output_dir=args.output_dir,
    )
