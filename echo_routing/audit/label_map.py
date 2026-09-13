"""Label ontology: raw EV9V codes -> versioned coarse families.

Raw codes are never rewritten. Mapping to families is explicit and versioned so any
anatomical claim can be traced back to `mapping_version` (proposal section 4.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from echo_routing.errors import LabelError

from echo_routing.config import REPO_ROOT

DEFAULT_LABELS_YAML = REPO_ROOT / "configs" / "labels.yaml"


@dataclass(frozen=True)
class LabelOntology:
    mapping_version: str
    raw_codes: tuple[str, ...]
    family_of: Mapping[str, str | None]  # raw code -> family or None (excluded)
    family_classes: tuple[str, ...]
    raw9_classes: tuple[str, ...]

    def family(self, raw_code: str) -> str | None:
        if raw_code not in self.family_of:
            raise LabelError(f"unknown raw code: {raw_code!r}")
        return self.family_of[raw_code]

    def family_index(self, raw_code: str) -> int | None:
        fam = self.family(raw_code)
        return None if fam is None else self.family_classes.index(fam)

    def raw9_index(self, raw_code: str) -> int:
        if raw_code not in self.raw9_classes:
            raise LabelError(f"raw code not in raw9 table: {raw_code!r}")
        return self.raw9_classes.index(raw_code)

    def classes(self, task: str) -> tuple[str, ...]:
        if task == "family5":
            return self.family_classes
        if task == "raw9":
            return self.raw9_classes
        raise LabelError(f"unknown task: {task!r} (expected 'family5' or 'raw9')")

    def index(self, task: str, raw_code: str) -> int | None:
        return self.family_index(raw_code) if task == "family5" else self.raw9_index(raw_code)


def load_ontology(path: Path = DEFAULT_LABELS_YAML) -> LabelOntology:
    with Path(path).open() as fh:
        cfg = yaml.safe_load(fh)
    raw_codes = tuple(cfg["raw_codes"])
    family_of = dict(cfg["family5"])
    family_classes = tuple(cfg["family5_classes"])
    raw9 = tuple(cfg["raw9_classes"])
    _validate(raw_codes, family_of, family_classes, raw9)
    return LabelOntology(
        mapping_version=str(cfg["mapping_version"]),
        raw_codes=raw_codes,
        family_of=family_of,
        family_classes=family_classes,
        raw9_classes=raw9,
    )


def _validate(raw_codes, family_of, family_classes, raw9) -> None:
    missing = set(raw_codes) - set(family_of)
    extra = set(family_of) - set(raw_codes)
    if missing or extra:
        raise LabelError(f"family5 mapping mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    bad = {c: f for c, f in family_of.items() if f is not None and f not in family_classes}
    if bad:
        raise LabelError(f"family values not in family5_classes: {bad}")
    if set(raw9) != set(raw_codes) or len(raw9) != len(raw_codes):
        raise LabelError("raw9_classes must be a permutation of raw_codes")
