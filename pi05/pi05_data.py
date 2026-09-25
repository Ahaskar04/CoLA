"""Hidden-marker handover demos in pi0.5's input format, one row per timestep.

Uses CoLA's episodes, splits and chunk starts. Per arm and timestep:
  image   : that arm's wrist camera, resize_with_pad to 224
  state   : 7-d joints + gripper, quantile-normalised, tokenised into the prompt
  actions : absolute 7-d joint targets, quantile-normalised, zero-padded to 32
An arm's row never contains its partner's observation. Derived arrays are
cached as .npy.
"""

import json
import pathlib

import h5py
import numpy as np

import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize

_DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
MANIFEST = str(_DATA / "cache" / "handover_marker" / "split_manifest.json")
CACHE = _DATA / "cache" / "handover_marker_pi05"
CHUNK = 10
ACTION_DIM = 32
STATE_RAW = 7
PROMPT = {
    "a": "hand the box to the other arm",
    "b": "take the box from the other arm and put it in a tray",
}
H5_KEYS = {"a": ("image_wrist_a", "state_a", "action_a"), "b": ("image_wrist_b", "state_b", "action_b")}


def resize_224(images_uint8: np.ndarray, batch: int = 256) -> np.ndarray:
    """openpi's resize_with_pad, the same call used at evaluation."""
    from openpi.shared import image_tools

    out = np.empty((len(images_uint8), 224, 224, 3), np.uint8)
    for i in range(0, len(images_uint8), batch):
        out[i:i + batch] = np.asarray(image_tools.resize_with_pad(images_uint8[i:i + batch], 224, 224))
    return out


def quantile_norm(x, stats):
    q01, q99 = np.asarray(stats.q01)[: x.shape[-1]], np.asarray(stats.q99)[: x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def quantile_unnorm(x, stats):
    q01, q99 = np.asarray(stats.q01)[: x.shape[-1]], np.asarray(stats.q99)[: x.shape[-1]]
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def valid_starts(lens: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """CoLA's _build_valid_start_indices."""
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    counts = np.clip(lens - chunk + 1, 0, None)
    return np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]).astype(np.int64)


def norm_stats_path(cache=CACHE):
    return pathlib.Path(cache) / "norm_stats.json"


def load_norm_stats(cache=CACHE):
    return _normalize.deserialize_json(norm_stats_path(cache).read_text())


def build_cache(manifest=MANIFEST, cache=CACHE, max_len=64):
    """Read the HDF5 episodes once and write every derived array. Idempotent."""
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    done = cache / "DONE.json"
    m = json.load(open(manifest))
    if done.exists():
        prov = json.loads(done.read_text())
        assert prov["manifest"] == str(manifest) and prov["max_len"] == max_len, (
            f"{cache} was built from {prov}; use a fresh cache dir")
        return

    raw = {}
    for split, paths in m["splits"].items():
        cols = {k: [] for k in ("img_a", "img_b", "state_a", "state_b", "act_a", "act_b")}
        lens, colours = [], []
        for p in paths:
            with h5py.File(p, "r") as f:
                for arm in "ab":
                    ik, sk, ak = H5_KEYS[arm]
                    cols[f"img_{arm}"].append(resize_224(f[ik][:]))
                    cols[f"state_{arm}"].append(f[sk][:].astype(np.float32))
                    cols[f"act_{arm}"].append(f[ak][:].astype(np.float32))
                lens.append(len(f["action_a"]))
                c = f.attrs["marker_color"]
                colours.append(c.decode() if isinstance(c, bytes) else str(c))
        raw[split] = {k: np.concatenate(v) for k, v in cols.items()}
        raw[split]["lens"] = np.asarray(lens, np.int64)
        raw[split]["colours"] = colours
        print(f"  [{split}] {len(paths)} episodes, {int(np.sum(lens))} timesteps, "
              f"{len(valid_starts(raw[split]['lens']))} chunk-starts")

    # Statistics from TRAIN only, per arm, with openpi's own RunningStats.
    stats = {}
    for key in ("state_a", "state_b", "act_a", "act_b"):
        rs = _normalize.RunningStats()
        rs.update(raw["train"][key])
        stats[key] = rs.get_statistics()
    _normalize.save(cache, stats)          # writes norm_stats.json

    tok = _tokenizer.PaligemmaTokenizer(max_len)
    for split, d in raw.items():
        out = {"lens": d["lens"]}
        for arm in "ab":
            s_norm = quantile_norm(d[f"state_{arm}"], stats[f"state_{arm}"])
            toks, masks = zip(*(tok.tokenize(PROMPT[arm], s) for s in s_norm))
            out[f"img_{arm}"] = d[f"img_{arm}"]
            out[f"state_{arm}"] = np.pad(s_norm, ((0, 0), (0, ACTION_DIM - STATE_RAW))).astype(np.float32)
            out[f"tok_{arm}"] = np.stack(toks).astype(np.int32)
            out[f"tokmask_{arm}"] = np.stack(masks).astype(bool)
            out[f"act_{arm}"] = np.pad(quantile_norm(d[f"act_{arm}"], stats[f"act_{arm}"]),
                                       ((0, 0), (0, ACTION_DIM - STATE_RAW))).astype(np.float32)
            longest = int(out[f"tokmask_{arm}"].sum(1).max())
            assert longest < max_len, f"prompt+state reaches {longest} tokens; raise max_len"
            print(f"  [{split}] arm {arm}: longest prompt {longest}/{max_len} tokens")
        for k, v in out.items():
            np.save(cache / f"{split}_{k}.npy", v)
        (cache / f"{split}_colours.json").write_text(json.dumps(d["colours"]))
    done.write_text(json.dumps({"manifest": str(manifest), "max_len": max_len, "prompts": PROMPT}))
    print(f"cache written -> {cache}")


class MarkerSplit:
    """Memory-mapped split with CoLA's epoch sampler, deterministic in the step
    index, so a resumed run draws the same batches as an uninterrupted one."""

    def __init__(self, split, cache=CACHE, in_memory=True):
        # Loaded into memory by default (memory-mapped random reads were slow).
        cache = pathlib.Path(cache)
        load = lambda k: np.load(cache / f"{split}_{k}.npy", mmap_mode=None if in_memory else "r")  # noqa: E731
        self.arrays = {f"{k}_{arm}": load(f"{k}_{arm}") for k in ("img", "state", "tok", "tokmask", "act")
                       for arm in "ab"}
        self.lens = np.load(cache / f"{split}_lens.npy")
        self.starts = valid_starts(self.lens)
        self.offsets = np.arange(CHUNK)

    def __len__(self):
        return len(self.starts)

    def steps_per_epoch(self, batch):
        return len(self.starts) // batch

    def indices_for_step(self, step, batch, seed=0):
        """CoLA's shuffled epochs with drop_last: step -> the chunk starts of that batch."""
        spe = self.steps_per_epoch(batch)
        epoch, k = divmod(step, spe)
        order = np.random.default_rng([seed, epoch]).permutation(len(self.starts))
        return self.starts[np.sort(order[k * batch:(k + 1) * batch])]

    def batch(self, starts):
        """Numpy batch for both arms: observation dicts and 10-step action chunks."""
        idx = starts[:, None] + self.offsets[None, :]
        out = {}
        for arm in "ab":
            A = self.arrays
            out[f"obs_{arm}"] = {
                "image": {"wrist_0_rgb": np.asarray(A[f"img_{arm}"][starts])},
                "image_mask": {"wrist_0_rgb": np.ones(len(starts), bool)},
                "state": np.asarray(A[f"state_{arm}"][starts]),
                "tokenized_prompt": np.asarray(A[f"tok_{arm}"][starts]),
                "tokenized_prompt_mask": np.asarray(A[f"tokmask_{arm}"][starts]),
            }
            out[f"actions_{arm}"] = np.asarray(A[f"act_{arm}"][idx])
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--max-len", type=int, default=64)
    a = ap.parse_args()
    build_cache(a.manifest, a.cache, a.max_len)
    tr = MarkerSplit("train", a.cache)
    print(f"train: {len(tr)} chunk-starts, {tr.steps_per_epoch(128)} steps/epoch at batch 128 "
          f"-> 150 epochs = {150 * tr.steps_per_epoch(128)} steps")
