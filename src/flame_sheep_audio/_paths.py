"""XDG base-directory resolution for flame-sheep's config and data dirs.

Centralizes path resolution so the call sites don't hardcode ~/.config and
~/.local/share. Uses platformdirs, so $XDG_CONFIG_HOME / $XDG_DATA_HOME are
honored when set and the defaults are unchanged otherwise (on Linux:
~/.config/flame-sheep and ~/.local/share/flame-sheep).

The appname is "flame-sheep" and MUST match across every package (they share
one config/data tree). This module is intentionally duplicated per package
rather than shared: the packages form a DAG and a shared paths module would
introduce a new cross-package dependency.
"""

import platformdirs
from pathlib import Path

_APP = "flame-sheep"


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir(_APP))


def data_dir() -> Path:
    return Path(platformdirs.user_data_dir(_APP))
