#!/usr/bin/env python3
"""Is the marker colour linearly decodable from CoLA's 64-d message?

Extracts msg_a = encoder_a(self_a) for every cached timestep of the
hidden-marker task and fits a multinomial logistic regression from the message
to the episode's marker colour. Chance is 33.3%.

WHY THIS TASK. The marker is on the box face turned toward arm A and is
occluded from B for the whole episode, so the colour is the one thing B cannot
see and must receive. If the 81.8% correct-tray result is real, the colour has
to be in the message.

THREE PROBES, NOT ONE. A probe on the message alone is not interpretable on its
own, so this runs all three and prints them together:

  self_a  -- A's own representation, the encoder's INPUT. The ceiling: colour
             the encoder never saw cannot be transmitted. If this is at chance
             the whole analysis stops here and the 81.8% needs another
             explanation.
  msg_a   -- the 64-d message. The number of interest.
  msg_a from the NO-MESSAGE checkpoint -- the control. That model's encoder
             still exists and still takes gradients; only its output is zeroed
             before the decoder reads it. If colour is decodable there too,
             then decodability is a property of the features, NOT evidence that
             the channel carries colour to B on purpose.

WHAT THIS DOES AND DOES NOT SHOW. A high score is positive SIGNALLING: the
information is present and linearly readable. It is NOT positive LISTENING --
it says nothing about whether B's policy uses it. Only an intervention (feed B
a message recorded from a different colour and see where the box lands) settles
that. Lowe et al., AAMAS 2019.

TIMESTEP CAVEAT. Every frame of an episode inherits that episode's colour
label, so frames from one episode are not independent samples. Splitting
frames at random would put neighbouring frames of the same episode on both
sides and inflate the score toward 100% by memorising episodes. This script
therefore splits BY EPISODE, and additionally reports a per-episode majority
vote, which is the number that matches how the policy is actually scored.

Runs on CPU in minutes; no GPU, no rollouts, no SU.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = "/scratch/users/ntu/ahaskar0/v1/cola-research-code"
sys.path.insert(0, f"{REPO}/handover/cola")

CACHE = "/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/cache_marker_v1"
FEATS = "/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/features_marker_v1"
MSG_CKPT = (f"{REPO}/handover/cola/experiments/run_marker_wristmsg"
            "/checkpoints/best_model.pkl")
NOMSG_CKPT = (f"{REPO}/handover/cola/experiments/run_marker_wristnomsg"
              "/checkpoints/best_model.pkl")
COLORS = ("blue", "green", "yellow")


def episode_colors(cache_dir, split):
    """Marker colour per episode, in the order the cache stored them."""
    import h5py
    manifest = json.load(open(pathlib.Path(cache_dir) / "split_manifest.json"))
    out = []
    for p in manifest["splits"][split]:
        with h5py.File(p, "r") as f:
            c = f.attrs.get("marker_color")
            if c is None:
                raise SystemExit(f"{p} has no marker_color attribute")
            out.append(c.decode() if isinstance(c, bytes) else str(c))
    return out


def load_split(cache_dir, feat_dir, split):
    """Per-frame features and states, plus the episode index of each frame."""
    lens = np.load(pathlib.Path(cache_dir) / f"{split}_episode_lens.npy")
    feat = np.load(pathlib.Path(feat_dir) / f"{split}_features_a.npy")
    state = np.load(pathlib.Path(cache_dir) / f"{split}_states_a.npy")
    if feat.shape[0] != lens.sum():
        raise SystemExit(
            f"{split}: {feat.shape[0]} feature rows but episode_lens sums to "
            f"{lens.sum()} -- cache and features are from different runs")
    ep_idx = np.repeat(np.arange(len(lens)), lens)
    return feat, state, ep_idx, lens


def normalise_states(cache_dir, states):
    """Same normalisation the trainer applied; the encoder expects it."""
    stats = json.load(open(pathlib.Path(cache_dir) / "state_stats.json"))
    s = stats["state_a"] if "state_a" in stats else stats["a"]
    mean = np.asarray(s["mean"], dtype=np.float32)
    std = np.asarray(s["std"], dtype=np.float32)
    return (states - mean) / np.maximum(std, 1e-6)


_MODEL_CACHE = {}


def build_messages(ckpt_path, feat, state_n, feat_o=None):
    """msg_a = encoder_a(self_a), and self_a itself, for every frame.

    COLAModel.__init__ loads the frozen Octo backbone (~90 s on CPU) even
    though nothing here needs it: _self_repr and encoder_a are pure functions
    of the checkpoint's params and the cached features. Models are therefore
    cached per (use_proprio, use_overhead, ...) shape so the backbone loads
    once per distinct configuration rather than once per call.
    """
    import pickle
    import jax.numpy as jnp
    import cola_architecture as CA

    ckpt = pickle.load(open(ckpt_path, "rb"))
    cfg = ckpt.get("config", {})
    params = ckpt["params"]
    use_proprio = bool(ckpt.get("use_proprio", cfg.get("use_proprio", False)))
    use_overhead = bool(ckpt.get("use_overhead", cfg.get("use_overhead", False)))

    model = CA.COLAModel(
        use_proprio=use_proprio, use_overhead=use_overhead,
        split_gripper=bool(ckpt.get("split_gripper", True)),
        use_diffusion=bool(cfg.get("use_diffusion", False)),
        diffusion_unet=bool(cfg.get("diffusion_unet", False)),
        unet_dims=tuple(cfg["unet_dims"]) if cfg.get("unet_dims") else None,
    )

    if use_overhead and feat_o is None:
        raise SystemExit("checkpoint uses the overhead view but no "
                         "*_features_o.npy was loaded")

    # Chunked: 150 episodes of features at 768-d is large enough that one
    # device_put of the whole split is wasteful.
    selves, msgs = [], []
    B = 4096
    for i in range(0, feat.shape[0], B):
        f = jnp.asarray(feat[i:i + B])
        s = jnp.asarray(state_n[i:i + B]) if use_proprio else None
        o = jnp.asarray(feat_o[i:i + B]) if use_overhead else None
        self_a = model._self_repr(f, s, params, "a", o)
        msg_a = model.encoder_a.apply(params["encoder_a"], self_a)
        selves.append(np.asarray(self_a))
        msgs.append(np.asarray(msg_a))
    return np.concatenate(selves), np.concatenate(msgs), {
        "use_proprio": use_proprio, "use_overhead": use_overhead,
        "use_messages": ckpt.get("use_messages"), "epoch": ckpt.get("epoch"),
    }


def probe_cv(X, y, ep, n_folds=5, seed=0):
    """Grouped k-fold over ALL episodes; every episode is held out exactly once.

    The fixed train/test split leaves only 15 test episodes (3 blue / 5 green /
    7 yellow), where one episode is 6.7% of the episode-vote score and the
    binomial spread on 15 draws swamps any difference from the 33.3% chance
    rate. Cross-validating over all 150 episodes tests each one while still
    training on ~120, so the estimate has 10x the evaluation sample at the same
    training size.

    Folds are grouped BY EPISODE, never by frame. Frames within an episode all
    carry that episode's label and are highly correlated, so a frame-level
    split would put neighbours on both sides and let the probe memorise
    episodes -- inflating the score toward 100% regardless of what the message
    encodes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    # Stratified so each fold keeps the colour balance; grouped so an episode's
    # frames never straddle a fold boundary.
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    frame_hit = frame_tot = 0
    ep_hit = ep_tot = 0
    per_fold = []

    for tr, te in skf.split(X, y, groups=ep):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))

        frame_hit += int((pred == y[te]).sum())
        frame_tot += len(te)

        fold_hit = fold_tot = 0
        for e in np.unique(ep[te]):
            m = ep[te] == e
            vote = np.bincount(pred[m], minlength=len(COLORS)).argmax()
            fold_hit += int(vote == y[te][m][0])
            fold_tot += 1
        ep_hit += fold_hit
        ep_tot += fold_tot
        per_fold.append(100.0 * fold_hit / max(fold_tot, 1))

    return (frame_hit / max(frame_tot, 1), ep_hit / max(ep_tot, 1),
            ep_hit, ep_tot, per_fold)


def probe(X_tr, y_tr, ep_tr, X_te, y_te, ep_te, seed=0):
    """Multinomial logistic regression; frame accuracy and episode-vote accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_tr)
    # No multi_class= argument: it was removed in scikit-learn 1.8, where
    # multinomial is the behaviour for a multi-class target anyway.
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(X_tr), y_tr)

    pred = clf.predict(sc.transform(X_te))
    frame_acc = float((pred == y_te).mean())

    # Per-episode majority vote: the policy commits to ONE tray per episode, so
    # this is the number that corresponds to how the task is scored.
    ep_correct = ep_total = 0
    for e in np.unique(ep_te):
        m = ep_te == e
        vote = np.bincount(pred[m], minlength=len(COLORS)).argmax()
        ep_correct += int(vote == y_te[m][0])
        ep_total += 1
    return frame_acc, ep_correct / max(ep_total, 1), ep_correct, ep_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--feat-dir", default=FEATS)
    ap.add_argument("--msg-ckpt", default=MSG_CKPT)
    ap.add_argument("--nomsg-ckpt", default=NOMSG_CKPT,
                    help="control: its encoder trained but its output was zeroed")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--cv", type=int, default=0, metavar="K",
                    help="pool train+val+test and run K-fold grouped CV "
                         "instead of the fixed split. The fixed test set "
                         "is 15 episodes, where one episode is 6.7%% of the "
                         "episode-vote score -- far too coarse to separate "
                         "a weak signal from the 33.3%% chance rate. K=5 "
                         "tests all 150 while training on ~120 each fold.")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    a = ap.parse_args()

    print("=" * 74)
    print("PROBE: is the marker colour linearly decodable from the message?")
    print("=" * 74)

    splits = {}
    for split in (a.train_split, a.test_split):
        feat, state, ep_idx, lens = load_split(a.cache_dir, a.feat_dir, split)
        cols = episode_colors(a.cache_dir, split)
        if len(cols) != len(lens):
            raise SystemExit(f"{split}: {len(cols)} episodes in the manifest but "
                             f"{len(lens)} in episode_lens")
        y = np.asarray([COLORS.index(cols[e]) for e in ep_idx])
        fo = pathlib.Path(a.feat_dir) / f"{split}_features_o.npy"
        splits[split] = {
            "feat": feat, "state_n": normalise_states(a.cache_dir, state),
            "feat_o": np.load(fo) if fo.exists() else None,
            "y": y, "ep": ep_idx, "n_ep": len(lens),
        }
        from collections import Counter
        print(f"  {split:5s}: {len(lens):3d} episodes, {feat.shape[0]:6d} frames, "
              f"colours {dict(Counter(cols))}")

    tr, te = splits[a.train_split], splits[a.test_split]
    results = {"chance": 100.0 / len(COLORS), "probes": {}}

    if a.cv:
        # Episode indices are per-split, so offset each split's before pooling
        # or episodes from different splits would share a group id and the
        # grouped fold would leak across them.
        pooled, off = {}, 0
        for split in ("train", "val", "test"):
            if split not in splits:
                feat, state, ep_idx, lens = load_split(a.cache_dir, a.feat_dir, split)
                cols = episode_colors(a.cache_dir, split)
                fo = pathlib.Path(a.feat_dir) / f"{split}_features_o.npy"
                splits[split] = {
                    "feat": feat, "state_n": normalise_states(a.cache_dir, state),
                    "feat_o": np.load(fo) if fo.exists() else None,
                    "y": np.asarray([COLORS.index(cols[e]) for e in ep_idx]),
                    "ep": ep_idx, "n_ep": len(lens),
                }
            d = splits[split]
            for k in ("feat", "state_n", "y"):
                pooled.setdefault(k, []).append(d[k])
            if d["feat_o"] is not None:
                pooled.setdefault("feat_o", []).append(d["feat_o"])
            pooled.setdefault("ep", []).append(d["ep"] + off)
            off += d["n_ep"]
        pooled = {k: np.concatenate(v) for k, v in pooled.items()}
        results["cv_folds"] = a.cv
        results["cv_episodes"] = int(off)
        from collections import Counter
        print(f"\n  POOLED for {a.cv}-fold CV: {off} episodes, "
              f"{pooled['feat'].shape[0]} frames, "
              f"colours {dict(Counter(np.asarray(COLORS)[pooled['y']]))}")

    def run(name, Xtr, Xte, note, Xcv=None):
        if a.cv:
            fa, ea, ec, et, folds = probe_cv(Xcv, pooled["y"], pooled["ep"],
                                             n_folds=a.cv)
            spread = f" | folds {' '.join(f'{f:.0f}' for f in folds)}"
        else:
            fa, ea, ec, et = probe(Xtr, tr["y"], tr["ep"], Xte, te["y"], te["ep"])
            folds, spread = None, ""
        print(f"\n{name}")
        print(f"   dim {(Xcv if a.cv else Xtr).shape[1]:5d} | frame {100*fa:5.1f}% | "
              f"episode-vote {100*ea:5.1f}% ({ec}/{et}) | chance 33.3%{spread}")
        print(f"   {note}")
        results["probes"][name] = {
            "dim": int((Xcv if a.cv else Xtr).shape[1]), "frame_acc": 100 * fa,
            "episode_acc": 100 * ea, "episodes_correct": ec, "episodes_total": et,
            "per_fold_episode_acc": folds,
        }

    print("\n--- messages-ON checkpoint ---")
    self_tr, msg_tr, meta = build_messages(a.msg_ckpt, tr["feat"], tr["state_n"], tr["feat_o"])
    self_te, msg_te, _ = build_messages(a.msg_ckpt, te["feat"], te["state_n"], te["feat_o"])
    self_cv = msg_cv = None
    if a.cv:
        self_cv, msg_cv, _ = build_messages(
            a.msg_ckpt, pooled["feat"], pooled["state_n"], pooled.get("feat_o"))
    print(f"   {meta}")
    results["msg_ckpt_meta"] = meta

    run("self_a  (encoder INPUT, the ceiling)", self_tr, self_te,
        "at chance here -> the colour never reaches the encoder at all",
        Xcv=self_cv)
    run("msg_a   (the 64-d message)", msg_tr, msg_te,
        "the number of interest: is the colour in what B receives?",
        Xcv=msg_cv)

    if a.nomsg_ckpt and pathlib.Path(a.nomsg_ckpt).exists():
        print("\n--- no-message checkpoint (CONTROL) ---")
        _, nmsg_tr, nmeta = build_messages(a.nomsg_ckpt, tr["feat"], tr["state_n"], tr["feat_o"])
        _, nmsg_te, _ = build_messages(a.nomsg_ckpt, te["feat"], te["state_n"], te["feat_o"])
        nmsg_cv = None
        if a.cv:
            _, nmsg_cv, _ = build_messages(
                a.nomsg_ckpt, pooled["feat"], pooled["state_n"], pooled.get("feat_o"))
        print(f"   {nmeta}")
        if nmeta.get("use_messages") is not False:
            print("   WARNING: this checkpoint does not record use_messages=False")
        results["nomsg_ckpt_meta"] = nmeta
        run("msg_a   (no-message model's encoder)", nmsg_tr, nmsg_te,
            "decodable here too -> decodability is a property of the features, "
            "not evidence the channel is used",
            Xcv=nmsg_cv)

    print("\n" + "=" * 74)
    print("Reading: high msg_a accuracy is positive SIGNALLING -- the colour is")
    print("present and linearly readable. It is NOT evidence that B's policy")
    print("uses it. Only a message-swap intervention shows positive LISTENING.")
    print("=" * 74)

    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
