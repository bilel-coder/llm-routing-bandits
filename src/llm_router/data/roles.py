"""
Partition-role loader.

The physical splits.json uses train/val/test naming. After the M2.5 audit
we reassigned semantic roles to keep one partition untouched for final
evaluation:

    TRAIN      → physical 'train'
    DEV        → physical 'test'   (used for pilots, tuning, plotting)
    FINAL_TEST → physical 'val'    (frozen; single final evaluation only)

Use `partition_for(role)` in downstream code so scripts read from role names
instead of physical partition names, and cannot accidentally load FINAL_TEST
without explicit authorisation.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROLES_PATH = Path("data/interim/partition_roles.json")

VALID_ROLES = {"TRAIN", "DEV", "FINAL_TEST"}


def _load_map() -> dict[str, str]:
    with open(_ROLES_PATH) as f:
        cfg = json.load(f)
    return cfg["role_to_partition"]


def partition_for(role: str, *, allow_final_test: bool = False) -> str:
    """
    Return the physical partition name for a semantic role.

    Parameters
    ----------
    role              : one of {'TRAIN', 'DEV', 'FINAL_TEST'}.
    allow_final_test  : must be True to resolve FINAL_TEST. Guards against
                        accidental exposure during development.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role '{role}'. Choose from {VALID_ROLES}.")
    if role == "FINAL_TEST" and not allow_final_test:
        raise RuntimeError(
            "Refusing to load FINAL_TEST without allow_final_test=True. "
            "FINAL_TEST is reserved for a single frozen final evaluation "
            "of A* only. Explicitly pass allow_final_test=True if you are "
            "running the final A* evaluation."
        )
    return _load_map()[role]


def load_role_map() -> dict[str, str]:
    """Full role→partition mapping (read-only view)."""
    return dict(_load_map())
