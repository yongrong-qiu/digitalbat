"""Single-toggle machine/path configuration for the analysis notebooks.

Usage (first cell of a notebook, cwd = ``notebooks/``)::

    import machine_paths as mp
    P = mp.setup()            # auto-detects the machine; or mp.setup('win_batlab')
    data_config = make_data_config(session_names=..., parent_folder=P.data_root)

``MACHINE = 'auto'`` picks the profile by OS; set it to a profile name to force one.
Every profile carries the three roots the notebooks need:

* ``data_root``  – parent folder holding ``Dataset_*`` session folders (trailing slash;
                   ``digitalbat.datasets.data`` concatenates with ``+``, so keep forward
                   slashes — they work on Windows too).
* ``code_root``  – folder that contains the ``digitalbat`` repo checkout.
* ``cebra_dir``  – local CEBRA checkout to put on ``sys.path``; ``None`` means use the
                   pip-installed ``cebra`` package from the active environment.
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# >>> THE ONE TOGGLE <<<   'auto' | 'ubuntu' | 'win_batlab'
MACHINE = 'auto'
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    name: str
    data_root: str          # e.g. 'S:/users/Adam/Angelo/NP_datasets/'
    code_root: str          # folder containing the digitalbat repo
    cebra_dir: str | None   # local CEBRA checkout, or None -> pip-installed cebra

    @property
    def unitmatch_root(self) -> Path:
        return Path(self.data_root) / 'UnitMatch'

    @property
    def repo_root(self) -> str:
        return os.path.join(self.code_root, 'digitalbat')


PROFILES = {
    # Original development machine (Ubuntu; network share mounted at /mnt/s5adam).
    'ubuntu': Paths(
        name='ubuntu',
        data_root='/mnt/s5adam/Angelo/NP_datasets/',
        code_root=os.path.expanduser('~/Documents/code'),
        cebra_dir=os.path.expanduser('~/Documents/code/CEBRA'),
    ),
    # Windows lab workstation (share mapped to S:\users\Adam; repo under Desktop\Adam).
    'win_batlab': Paths(
        name='win_batlab',
        data_root='S:/users/Adam/Angelo/NP_datasets/',
        code_root=os.path.expanduser('~/Desktop/Adam'),
        cebra_dir=None,
    ),
}


def detect() -> str:
    if MACHINE != 'auto':
        return MACHINE
    return 'win_batlab' if platform.system() == 'Windows' else 'ubuntu'


def setup(machine: str | None = None, verbose: bool = True) -> Paths:
    """Resolve the profile, put the repo (and optional CEBRA checkout) on ``sys.path``."""
    P = PROFILES[machine or detect()]
    # Prefer the checkout this notebook lives in; fall back to the profile's code_root.
    here_repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for p in (here_repo, P.repo_root):
        if os.path.isdir(os.path.join(p, 'digitalbat')) and p not in sys.path:
            sys.path.append(p)
    if P.cebra_dir and os.path.isdir(P.cebra_dir):
        sys.path.insert(0, P.cebra_dir)
    if verbose:
        ok = os.path.isdir(P.data_root)
        print(f'[machine_paths] profile={P.name}  data_root={P.data_root} '
              f'({"found" if ok else "MISSING"})  cebra={P.cebra_dir or "pip"}')
    return P
