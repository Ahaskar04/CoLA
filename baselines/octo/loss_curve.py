"""Per-run loss curve, written into the checkpoint folder next to the weights.

Used by finetune_octo.py during training.
"""

import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:            # don't let a plotting error kill the run
    HAS_MPL = False


def _title(save_dir, meta):
    meta = meta or {}
    bits = [Path(save_dir).name]
    if meta.get('arm'):
        bits.append(f"arm {meta['arm']}")
    if meta.get('head'):
        bits.append(f"head {meta['head']}")
    if 'use_proprio' in meta:
        bits.append('proprio' if meta['use_proprio'] else 'no proprio')
    if meta.get('camera'):
        bits.append(meta['camera'])
    return '  |  '.join(bits)


def write_history(history, save_dir, meta=None):
    """Save training_history.json and loss_curve.png in save_dir.

    Called after every validation, so an interrupted run still leaves a curve.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history = sorted(history, key=lambda h: h['step'])
    json.dump(history, open(save_dir / 'training_history.json', 'w'), indent=2)
    if not (HAS_MPL and history):
        return None

    steps = [h['step'] for h in history]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(steps, [h['train_loss'] for h in history], 'C0', lw=1.8,
            marker='o', ms=3, label='train')
    ax.plot(steps, [h['val_loss'] for h in history], 'C1', lw=1.8, ls='--',
            marker='s', ms=3, label='val')
    best = min(history, key=lambda h: h['val_loss'])
    ax.axvline(best['step'], color='r', ls=':', alpha=0.7,
               label=f"best val {best['val_loss']:.4f} @ step {best['step']}")
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_title(_title(save_dir, meta), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = save_dir / 'loss_curve.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png
