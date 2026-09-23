"""One table over every Octo baseline evaluation on disk.

Joins three sources that were written at different times and never shared a
schema: the eval results.json files, each checkpoint's finetune_meta.json /
training_history.json, and the "finetune arm=.. steps=.. batch=.." banner in
the PBS logs (the only record of steps and batch size).

Fields the older results.json files never recorded -- handover-only start,
gripper snapping, which pretrained model -- are recovered from the run tag and
marked as derived in the notes below, not silently invented.

    python collect_results.py                 # print, and write csv + md
"""

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path('/home/users/ntu/ahaskar0/CoLA/experiments/octo_baseline')
CKPT_ROOT = Path('/scratch/users/ntu/ahaskar0/octo_finetune')
LOG_DIR = Path('/home/users/ntu/ahaskar0/CoLA/baselines/octo_small/logs')

BANNER_RE = re.compile(
    r'finetune arm=(\S+)\s+steps=(\d+)\s+batch=(\d+)\s+-> +(\S+)')
# "   train: 130 episodes, 20170 transitions, ..." -- printed by H5Split at
# load time, and the only record of how big each run's training split was.
# One transition is one window start, so it is also one epoch's worth of
# samples: epochs = steps * batch / transitions.
TRANS_RE = re.compile(r'train: (\d+) episodes, (\d+) transitions')

# run tag -> (group, human-readable policy). Everything not listed is dropped,
# so a stray file cannot silently enter the table as an unexplained row.
GROUPS = {
    'v1 wrist camera': ['octo_ft_decentralised', 'octo_ft_decentralised_snap',
                        'octo_ft_ho_only', 'octo_ft_ho_only_snap'],
    'v1 third-person, 50 ep': ['tp_l1_prop', 'tp_diff_prop', 'tp_diff_noprop',
                               'tp_handover', 'tp_handover_only',
                               'tp_central', 'tp_central_ho'],
    'v1 third-person, 200 ep': ['tp_handover@f', 'tp_handover_only@f',
                                'tp_central@f', 'tp_central_ho@f'],
    'v2 third-person, 5k steps': ['v2_tp_handover', 'v2_tp_handover_only',
                                  'v2_tp_central', 'v2_tp_central_ho'],
    'v2 third-person, 15k steps': ['v2_15k_handover', 'v2_15k_handover_only',
                                   'v2_15k_central', 'v2_15k_central_ho'],
    'v2 diffusion, 12.6k steps': ['v2_diff_prop_handover',
                                 'v2_diff_prop_handover_only',
                                 'v2_diff_noprop_handover',
                                 'v2_diff_noprop_handover_only'],
    'CoLA-matched, full finetune': ['ho_full_decent', 'ho_full_central',
                                    'ft_full_decent', 'ft_full_central',
                                    'ft_full_decent_ho', 'ft_full_central_ho'],
    'CoLA-matched, head-only': ['ho_head_l1_decent', 'ho_head_l1_central',
                               'ho_head_diff_decent', 'ho_head_diff_central',
                               'ft_head_l1_decent', 'ft_head_l1_central',
                               'ft_head_l1_decent_ho', 'ft_head_l1_central_ho',
                               'ft_head_diff_decent', 'ft_head_diff_central',
                               'ft_head_diff_decent_ho',
                               'ft_head_diff_central_ho'],
    'zero-shot, 200 ep': ['zs_octo_small', 'zs_octo_small_ho',
                          'zs_octo_base', 'zs_octo_base_ho'],
    'zero-shot probes': ['octo_handover_wrist', 'octo_ctrl_frozen',
                         'octo_ctrl_random', 'octo_scale_0.25',
                         'octo_scale_0.5', 'octo_scale_1.0',
                         'octo_scale_2.0', 'octo_scale_4.0'],
}
ORDER = [(g, t) for g, tags in GROUPS.items() for t in tags]


def train_index():
    """checkpoint name -> {steps, batch, best_val, final_val, head, proprio}."""
    idx = {}
    for log in sorted(LOG_DIR.glob('*.OU')):
        text = log.read_text(errors='ignore')
        pieces = BANNER_RE.split(text)
        # split() yields [pre, arm, steps, batch, tag, body, arm, ...]
        for arm, steps, batch, tag, body in zip(*[pieces[i::5] for i in
                                                  range(1, 6)]):
            # A longer run supersedes a shorter one writing the same folder.
            if tag in idx and int(steps) <= idx[tag].get('steps', 0):
                continue
            e = {'arm': arm, 'steps': int(steps), 'batch': int(batch)}
            m = TRANS_RE.search(body)
            if m:
                e['episodes'], e['transitions'] = int(m[1]), int(m[2])
                e['epochs'] = e['steps'] * e['batch'] / e['transitions']
            idx[tag] = e
    for d in CKPT_ROOT.iterdir() if CKPT_ROOT.exists() else []:
        e = idx.setdefault(d.name, {})
        try:
            m = json.load(open(d / 'finetune_meta.json'))
            e.update(head=m.get('head'), proprio=m.get('use_proprio'),
                     camera=m.get('camera'), tune=m.get('tune'),
                     manifest=m.get('manifest'))
        except Exception:
            pass
        try:
            h = json.load(open(d / 'training_history.json'))
            if h:
                e['final_val'] = h[-1]['val_loss']
                e['best_val'] = min(p['val_loss'] for p in h)
                e['hist_steps'] = h[-1]['step']
        except Exception:
            pass
    return idx


def find_results():
    """tag -> path. '@f' marks the 200-episode folder copy of a 50-episode tag."""
    out = {}
    # logs/ first: those are the earlier 50-episode runs and own the bare tag.
    for p in sorted(ROOT.glob('logs/*.json')) + sorted(ROOT.glob('octo-*/logs/*.json')):
        out.setdefault(p.stem, p)
    # Per-run folders are the 200-episode reruns; they take '@f', and the bare
    # tag only when no 50-episode file exists (zs_*, v2_*).
    for p in sorted(ROOT.glob('*/results.json')):
        out[p.parent.name + '@f'] = p
        out.setdefault(p.parent.name, p)
    return out


def describe(tag, d, tinfo):
    """Everything the results.json did not record, recovered from the tag."""
    ck = d.get('checkpoints') or {}
    base = tag.replace('@f', '')
    # Prefer what the run recorded; fall back to the tag for older results
    # written before eval_octo.py stored the flag.
    handover_only = d.get('handover_only')
    if handover_only is None:
        handover_only = (base.endswith(('_ho', '_handover_only'))
                         or 'ho_only' in base or base.startswith('ho_'))
    snap = base.endswith('_snap')

    if d.get('action_stats') == 'bridge_dataset':
        if 'random' in base:
            policy, ckpts = 'random actions', []
        elif 'frozen' in base:
            policy, ckpts = 'frozen (zero action)', []
        elif 'base' in base or base == 'octo_handover_wrist':
            policy, ckpts = 'frozen octo-base', []
        else:
            policy, ckpts = 'frozen octo-small', []
        data = '-'
    elif ck.get('centralised'):
        policy, ckpts, data = 'centralised 14-D', [Path(ck['centralised']).name], None
    elif ck.get('a') and ck.get('b'):
        policy = 'decentralised 2x7-D'
        ckpts = [Path(ck['a']).name, Path(ck['b']).name]
        data = None
    else:
        # Pre-dates the checkpoints key: the tag is the checkpoint folder name.
        policy = 'single arm A (arm B held)'
        ckpts, data = [base], None

    if data is None:
        mans = {tinfo.get(c, {}).get('manifest') for c in ckpts}
        mans.discard(None)
        if mans:
            data = '+'.join(sorted(
                m.rstrip('/').split('/')[-2].replace('cache_aloha_handover', 'v')
                 .replace('cache_', '') for m in mans))
        else:
            data = 'v2' if any(c.startswith('v2') for c in ckpts) else 'v1'

    t = [tinfo.get(c, {}) for c in ckpts]
    steps = '/'.join(str(x.get('steps', '?')) for x in t) if t else '-'
    batch = '/'.join(str(x.get('batch', '?')) for x in t) if t else '-'
    val = '/'.join(f"{x['final_val']:.3f}" if 'final_val' in x else '?'
                   for x in t) if t else '-'
    head = '/'.join(str(x.get('head') or '?') for x in t) if t else '-'
    prop = '/'.join('y' if x.get('proprio') else 'n' for x in t) if t else '-'
    tune = '/'.join(str(x.get('tune') or 'full') for x in t) if t else '-'
    ep = [x['epochs'] for x in t if 'epochs' in x]
    tr = [x['transitions'] for x in t if 'transitions' in x]
    return dict(
        tune=tune,
        epochs=f'{max(ep):.1f}' if ep else '-',
        transitions=str(max(tr)) if tr else '-',policy=policy, data=data, ckpts=','.join(ckpts) or '-',
                steps=steps, batch=batch, final_val=val, head=head,
                proprio=prop, start='handover' if handover_only else 'full',
                snap='y' if snap else 'n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--out', default=str(ROOT))
    args = ap.parse_args()

    tinfo = train_index()
    found = find_results()
    rows, missing = [], []
    for group, tag in ORDER:
        p = found.get(tag)
        if p is None:
            missing.append(tag)
            continue
        d = json.load(open(p))
        s = d['summary']
        r = {'group': group, 'run': tag.replace('@f', '')}
        r.update(describe(tag, d, tinfo))
        r.update(
            camera=(d.get('arm_a') or {}).get('camera', '-'),
            arm_b=d.get('arm_b_mode', '-'),
            criterion=d.get('criterion', {}).get('name', '-'),
            n=s['n_scored'], skipped=d.get('skipped', 0),
            a_touch=s['a_touch_rate'], a_lift=s['a_lift_rate'],
            b_touch=s['b_touch_rate'], transfer=s['transfer_rate'],
            success=s['success_rate'], drop=s['drop_rate'],
            drop_given_transfer=s['drop_given_transfer'],
            mean_steps=s['mean_control_steps'],
            mean_transfer_step=s['mean_transfer_step'],
            min_a_box_dist_median=s['min_a_box_dist_median'],
            source=str(Path(p).relative_to(args.root)))
        rows.append(r)

    cols = list(rows[0].keys())
    out = Path(args.out)
    with open(out / 'all_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Per-column precision. A blanket %.1f turned a 66 mm approach distance
    # into "0.1", so distances are carried in millimetres and step counts are
    # whole numbers.
    DP = {'mean_steps': 0, 'mean_transfer_step': 0}

    def fmt(v, c=None):
        if v is None or v == '':
            return '-'
        if c == 'min_a_box_dist_median':
            return f'{float(v) * 1000:.1f}'
        if isinstance(v, float):
            return f'{v:.{DP.get(c, 1)}f}'
        return str(v)

    def table(keys, rs):
        head = ['a_box_dist_mm' if k == 'min_a_box_dist_median' else k
                for k in keys]
        return ('| ' + ' | '.join(head) + ' |\n'
                + '|' + '|'.join('---' for _ in keys) + '|\n'
                + '\n'.join('| ' + ' | '.join(fmt(r[k], k) for k in keys) + ' |'
                             for r in rs) + '\n')

    (out / 'all_results.md').write_text(table(cols, rows))

    # Notion converts a pasted markdown table into a real table block, but a
    # 30-column paste is unusable there, so the same rows are also emitted in
    # the three views the HTML page uses.
    VIEWS = {
        'outcome': ['group', 'run', 'policy', 'data', 'start', 'arm_b', 'n',
                    'a_touch', 'a_lift', 'b_touch', 'transfer', 'success',
                    'drop', 'drop_given_transfer'],
        'training': ['group', 'run', 'policy', 'data', 'ckpts', 'steps',
                     'batch', 'epochs', 'transitions', 'final_val', 'head',
                     'proprio', 'camera'],
        'dynamics': ['group', 'run', 'start', 'n', 'skipped', 'criterion',
                     'arm_b', 'snap', 'mean_steps', 'mean_transfer_step',
                     'min_a_box_dist_median'],
    }
    parts = ['# Octo handover baselines\n',
             f'{len(rows)} evaluation runs. Paste any table below straight into '
             'Notion; it converts to a table block.\n']
    for name, keys in VIEWS.items():
        parts += [f'\n## {name}\n', table(keys, rows)]
    parts += ['\n## everything\n', table(cols, rows)]
    (out / 'all_results_notion.md').write_text('\n'.join(parts))

    # Tab-separated, as a fallback paste target and for spreadsheets.
    with open(out / 'all_results.tsv', 'w') as f:
        f.write('\t'.join('a_box_dist_mm' if c == 'min_a_box_dist_median'
                           else c for c in cols) + '\n')
        for r in rows:
            f.write('\t'.join(fmt(r[c], c) for c in cols) + '\n')

    show = ['run', 'policy', 'data', 'start', 'arm_b', 'n', 'a_lift',
            'b_touch', 'transfer', 'success', 'drop']
    wid = {c: max(len(c), max(len(fmt(r[c], c)) for r in rows)) for c in show}
    last = None
    for r in rows:
        if r['group'] != last:
            last = r['group']
            print(f'\n--- {last} ' + '-' * (60 - len(last)))
            print('  '.join(c.rjust(wid[c]) if c not in ('run', 'policy', 'data',
                  'start', 'arm_b') else c.ljust(wid[c]) for c in show))
        print('  '.join(fmt(r[c], c).rjust(wid[c]) if c not in ('run', 'policy',
              'data', 'start', 'arm_b') else fmt(r[c], c).ljust(wid[c])
              for c in show))
    print(f'\n{len(rows)} runs -> {out/"all_results.csv"} and all_results.md')
    if missing:
        print(f'not on disk yet: {", ".join(missing)}')


if __name__ == '__main__':
    main()
