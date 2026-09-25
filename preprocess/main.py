"""Build a task's training cache and frozen Octo-Base features in one command.

    python preprocess/main.py handover_2arm
    python preprocess/main.py handover_marker
    python preprocess/main.py handover_3arm

Reads data/demos/<task> and writes data/cache/<task> and data/features/<task>,
running the task's scripts in order with the settings in TASKS. Flags override
the folders and settings; --steps runs only part of the pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Per task: the cache scripts (the three-arm one also caches joint states), the
# feature script, and the settings used for the paper.
TASKS = {
    'handover_2arm': {
        'cache': ['prepare_cache_2arm.py', 'prepare_states_2arm.py'],
        'features': 'extract_features_2arm.py',
        'overhead': True,
        'max_episodes': 0,
    },
    # Wrist cameras only: the marker is hidden from the overhead camera.
    'handover_marker': {
        'cache': ['prepare_cache_2arm.py', 'prepare_states_2arm.py'],
        'features': 'extract_features_2arm.py',
        'overhead': False,
        'max_episodes': 0,
    },
    'handover_3arm': {
        'cache': ['prepare_cache_3arm.py'],
        'features': 'extract_features_3arm.py',
        'overhead': True,
        'max_episodes': 130,
    },
}


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print(f'\n$ python preprocess/{script} {" ".join(map(str, args))}', flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task', choices=TASKS)
    ap.add_argument('--steps', nargs='+', choices=['cache', 'features'],
                    default=['cache', 'features'],
                    help='cache: splits, normalised actions and joint states; '
                         'features: Octo-Base features (needs a GPU)')
    ap.add_argument('--data_dir', type=Path, help='default: data/demos/<task>')
    ap.add_argument('--cache_dir', type=Path, help='default: data/cache/<task>')
    ap.add_argument('--feat_dir', type=Path, help='default: data/features/<task>')
    ap.add_argument('--overhead', action=argparse.BooleanOptionalAction, default=None,
                    help="also extract the overhead camera (default: the task's setting)")
    ap.add_argument('--max_episodes', type=int, default=None,
                    help="cap on episodes, sampled before the split (default: the "
                         "task's setting; 0 = all)")
    ap.add_argument('--seed', type=int, default=42,
                    help='seed for the episode cap and the 80/10/10 split')
    ap.add_argument('--batch_size', type=int, default=64,
                    help='images per Octo forward pass')
    args = ap.parse_args()

    task = TASKS[args.task]
    data_dir = args.data_dir or REPO / 'data' / 'demos' / args.task
    cache_dir = args.cache_dir or REPO / 'data' / 'cache' / args.task
    feat_dir = args.feat_dir or REPO / 'data' / 'features' / args.task
    overhead = task['overhead'] if args.overhead is None else args.overhead
    max_episodes = task['max_episodes'] if args.max_episodes is None else args.max_episodes

    if 'cache' in args.steps:
        prepare, *rest = task['cache']
        run(prepare, '--data_dir', data_dir, '--cache_dir', cache_dir,
            '--max_episodes', max_episodes, '--seed', args.seed)
        for script in rest:
            run(script, '--cache_dir', cache_dir)

    if 'features' in args.steps:
        run(task['features'], '--cache_dir', cache_dir, '--feat_dir', feat_dir,
            '--batch_size', args.batch_size, *(['--overhead'] if overhead else []))

    print(f'\nDone: {args.task} -> {cache_dir}, {feat_dir}')


if __name__ == '__main__':
    main()
