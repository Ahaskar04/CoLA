"""Make the split layout importable without editing the copied sources.

The files in this snapshot were written for a FLAT working directory: they do
`from cola_architecture import COLAModel`, `from probe_messages import ...`,
and so on, plus `sys.path.insert` calls naming absolute paths on the original
cluster. Splitting them into cola/ train/ eval/ analysis/ for readability
breaks every one of those imports.

Rather than rewrite twenty imports across code that is meant to be an
unmodified record of what was run, importing this module puts every source
directory on sys.path, which restores the flat namespace the files expect:

    import conftest_paths   # noqa: F401
    from cola_architecture import COLAModel

Alternatively, export the same set as PYTHONPATH before running anything:

    export PYTHONPATH=$(python3 conftest_paths.py):$PYTHONPATH
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
_DIRS = [
    _ROOT / "cola",
    _ROOT / "train",
    _ROOT / "train" / "prepare",
    _ROOT / "eval",
    _ROOT / "eval" / "ablations",
    _ROOT / "analysis",
    _ROOT / "baselines" / "octo",
    _ROOT / "pi05",
    _ROOT / "pi05" / "frozen",
    _ROOT / "envs" / "common",
]

for _d in _DIRS:
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

if __name__ == "__main__":
    print(":".join(str(d) for d in _DIRS if d.is_dir()))
