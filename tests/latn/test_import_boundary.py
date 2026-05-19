"""Enforced import boundary for the extractable LATN core (L1-5).

"Engraf tests still green" proves no regression; it does NOT prove the core
is decoupled. This test does: no module in the core may import the Engraf
application layer. As later seams land, entries are removed from
KNOWN_REMAINING and the boundary tightens automatically.
"""

import re
from pathlib import Path

import engraf

CORE_PACKAGES = ["lexer", "atn", "pos", "An_N_Space_Model", "utils"]
FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+engraf\.(visualizer|interpreter|llm_layer6)\b",
    re.MULTILINE,
)

# Couplings that later phase-1 seams remove. Each MUST cite its seam so this
# list shrinks deliberately, never silently grows.
KNOWN_REMAINING = {
    # seam #4: stale `SceneObject` import; unused on the L3 path (serves the
    # interpreter-only calculate_spatial_position). Removed when SpatialValidator
    # is parameterized.
    "utils/spatial_validation.py",
    # non-core example scripts, not library code; excluded at the physical split.
    "An_N_Space_Model/demo_scene_setup.py",
    "An_N_Space_Model/demo_sentence_interpreter.py",
}

ENGRAF_ROOT = Path(engraf.__file__).parent


def test_core_does_not_import_application_layer():
    offenders = []
    for pkg in CORE_PACKAGES:
        for path in (ENGRAF_ROOT / pkg).rglob("*.py"):
            rel = path.relative_to(ENGRAF_ROOT).as_posix()  # e.g. "utils/spatial_validation.py"
            if rel in KNOWN_REMAINING:
                continue
            m = FORBIDDEN.search(path.read_text())
            if m:
                offenders.append(f"{rel}: {m.group(0).strip()}")
    assert not offenders, "Core imports the application layer:\n" + "\n".join(offenders)


def test_known_remaining_entries_still_exist():
    """Guard against stale allowlist: every KNOWN_REMAINING file must exist
    and still contain a forbidden import, else delete the entry."""
    for rel in KNOWN_REMAINING:
        path = ENGRAF_ROOT / rel
        assert path.exists(), f"KNOWN_REMAINING stale (missing): {rel}"
        if not rel.startswith("An_N_Space_Model/demo_"):
            assert FORBIDDEN.search(path.read_text()), (
                f"KNOWN_REMAINING stale (no forbidden import left): {rel} "
                f"-- remove it; the boundary has tightened."
            )
