"""Official FACT (Lu and Elhamifar, CVPR 2024; ZijiaLewisLu/CVPR2024-FACT, pinned) as a temporal head on frozen
per-frame features (2026-10-08 FACT integration plan).

The authors' model (models/blocks.FACT), matching criterion (models/loss.MatchCriterion), null-class weight rule
(utils/train_tools.compute_null_weight) and configuration defaults (configs/default.py plus one of the official dataset
YAML files) are imported from the pinned checkout without modification. The adaptation is the data interface only.
- Training sequences come from the shared batch plan. They are unpadded and passed to the official forward one video
  at a time, as the official loader does, and the official loss is the mean over the update's videos.
- The dataset-specific settings follow the official guide (DATASET_SETUP_GUIDE.md, utils/gen_config.py). The token
  count is int(token_ratio x the longest training transcript), the average transcript length comes from the training
  sequences only, and there is no background class.
- Inference passes a constant dummy label sequence, never the reference labels. The official interface still takes
  labels, so the integration check verifies that predictions do not depend on them.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from echo_routing.temporal.dev_data import Seq
from echo_routing.temporal.dev_heads import check_pinned

FACT_PINNED = "7bd81bda2b84618a3e23876a2637a82f24881122"
PACKAGE = "fact_official"
MODULES = {"blocks": "models.blocks", "loss": "models.loss", "basic": "models.basic", "default": "configs.default",
           "train_tools": "utils.train_tools"}


def official(repo: Path) -> dict[str, Any]:
    """The pinned checkout imported as one package (its modules use relative imports), never added to sys.path."""
    repo = Path(repo).resolve()
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, repo / "__init__.py", submodule_search_locations=[str(repo)])
        mod = importlib.util.module_from_spec(spec); sys.modules[PACKAGE] = mod; spec.loader.exec_module(mod)
    elif Path(sys.modules[PACKAGE].__path__[0]).resolve() != repo:
        raise RuntimeError(f"{PACKAGE} is already imported from {sys.modules[PACKAGE].__path__[0]}, not {repo}")
    return {k: importlib.import_module(f"{PACKAGE}.{v}") for k, v in MODULES.items()}


def transcript_length(labels: np.ndarray) -> int:
    """Number of label runs, the official segment parsing (utils.parse_label): a same-class join is one segment."""
    labels = np.asarray(labels)
    return int(1 + np.count_nonzero(labels[1:] != labels[:-1])) if labels.size else 0


@dataclass(frozen=True)
class TokenRule:
    """The guide's rule for one-to-one matching, computed on the training sequences only."""
    max_transcript: int
    average_transcript: float
    token_ratio: float
    ntoken: int


def token_rule(stream_labels: list[np.ndarray], natives_per_epoch: int, token_ratio: float) -> TokenRule:
    """ntoken = int(token_ratio x the longest transcript) as in utils/gen_config.py. The average transcript length is
    taken over one epoch's training sequences: every constructed stream once plus `natives_per_epoch` native windows,
    each of which is one segment."""
    lens = [transcript_length(lab) for lab in stream_labels]
    if not lens or min(lens) < 1:
        raise ValueError("every training stream needs at least one sample")
    longest = max(max(lens), 1)
    average = (sum(lens) + int(natives_per_epoch)) / (len(lens) + int(natives_per_epoch))
    if token_ratio < 1:
        raise ValueError("the guide requires token_ratio >= 1")
    return TokenRule(longest, float(average), float(token_ratio), int(longest * token_ratio))


def fact_cfg(mods: dict, repo: Path, base_config: str, rule: TokenRule, n_classes: int):
    """configs/default.py merged with the official dataset YAML, then the dataset fields of utils/gen_config.py and the
    official null-class weight (utils/train_tools.compute_null_weight, which freezes the config)."""
    cfg = mods["default"].get_cfg_defaults(); cfg.merge_from_file(str(Path(repo) / "configs" / base_config))
    cfg.dataset = "ev9v_family5"; cfg.bg_class = []; cfg.average_transcript_len = rule.average_transcript
    cfg.FACT.ntoken = rule.ntoken; cfg.FACT.trans = False; cfg.Loss.match = "o2o"
    if float(cfg.Loss.nullw) != -1.0:
        raise ValueError("the base config must leave the null weight to the official rule (nullw -1)")
    mods["train_tools"].compute_null_weight(cfg, SimpleNamespace(average_transcript_len=rule.average_transcript, nclasses=n_classes))
    return cfg


class FACTHead:
    """Official FACT with build(), loss(model, batch) -> (loss, parts) and decode(model, seqs, device), the interface of
    the shared temporal heads (dev_heads)."""

    name = "fact"

    def __init__(self, repo: Path, in_dim: int, rule: TokenRule, num_classes: int = 5, base_config: str = "breakfast.yaml") -> None:
        self.commit = check_pinned(Path(repo), FACT_PINNED)
        self.mods = official(Path(repo)); self.rule, self.base_config = rule, base_config
        self.in_dim, self.num_classes = int(in_dim), int(num_classes)
        self.cfg = fact_cfg(self.mods, Path(repo), base_config, rule, self.num_classes)

    def build(self) -> nn.Module:
        model = self.mods["blocks"].FACT(self.cfg, self.in_dim, self.num_classes)
        model.mcriterion = self.mods["loss"].MatchCriterion(self.cfg, self.num_classes, [])   # official train.py; no background
        return model

    @staticmethod
    def unpad(batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The collated batch back to its videos: features (T_i, D) and labels (T_i,), without padding."""
        lengths = [int(n) for n in batch["lengths"].tolist()]
        seqs = [batch["x"][i, :, :n].T.contiguous() for i, n in enumerate(lengths)]
        labels = [batch["labels"][i, :n].contiguous() for i, n in enumerate(lengths)]
        if any(bool((lab < 0).any()) for lab in labels):
            raise ValueError("a padding label lies inside a sequence")
        return seqs, labels

    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        seqs, labels = self.unpad(batch)
        loss, _ = model(seqs, labels, compute_loss=True)    # official: the mean of the per-video losses
        return loss, {"fact": float(loss.detach())}

    @torch.no_grad()
    def predict(self, model: nn.Module, x: torch.Tensor) -> dict[str, np.ndarray]:
        """One video (T, D) through the official forward with a dummy label sequence, in eval mode."""
        dummy = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pred = model([x], [dummy])[0]["pred"]
        last = model.block_list[-1]
        frame_prob = torch.softmax(last.frame_clogit[:, 0], dim=-1).cpu().numpy()
        token_class = last.action_clogit[:, 0].argmax(-1).cpu().numpy()
        return {"labels": np.asarray(pred).astype(int), "frame_prob": frame_prob, "frame_labels": frame_prob.argmax(1),
                "tokens_used": int((token_class != self.num_classes).sum())}

    @torch.no_grad()
    def decode(self, model: nn.Module, seqs: list[Seq], device, batch_size: int = 1) -> dict[str, dict[str, Any]]:
        model.eval(); out = {}
        for s in seqs:
            out[s.seq_id] = self.predict(model, torch.from_numpy(s.feat.astype(np.float32)).to(device))
        return out


def loss_parts(head: FACTHead, model: nn.Module, label: torch.Tensor) -> dict[str, Any]:
    """The official per-block loss of one video split into its terms, after model._forward_one_video. Each block's
    weighted sum is checked against the official compute_loss of that block. Logging and checks only."""
    loss_m, basic = head.mods["loss"], head.mods["basic"]; blocks = head.mods["blocks"]
    crit = model.mcriterion; crit.set_label(label); sw = float(head.cfg.Loss.sw)
    last = model.block_list[-1]
    match = crit.match(basic.logit2prob(last.action_clogit, dim=-1), last.a2f_attn)
    per_block = []
    for b in model.block_list:
        t: dict[str, torch.Tensor] = {"frame": crit.frame_loss(b.frame_clogit.squeeze(1)), "token": crit.action_token_loss(match, b.action_clogit)}
        fl = loss_m.smooth_loss(torch.transpose(b.frame_clogit, 0, 1))
        if isinstance(b, blocks.InputBlock):
            t["smooth"] = fl; w = {"frame": 1.0, "token": 1.0, "smooth": sw}
            total = t["frame"] + t["token"] + sw * fl
        elif isinstance(b, blocks.UpdateBlockTDU):
            t["seg"] = crit.frame_loss_tdu(b.seg_clogit, b.tdu)
            t["f2a"] = crit.cross_attn_loss_tdu(match, torch.transpose(b.f2a_attn_logit, 1, 2), b.tdu, dim=1)
            t["a2f"] = crit.cross_attn_loss_tdu(match, b.a2f_attn_logit, b.tdu, dim=2)
            t["smooth"] = fl; w = {"frame": 0.5, "seg": 0.5, "token": 1.0, "f2a": 1.0, "a2f": 1.0, "smooth": sw}
            total = (t["frame"] + t["seg"]) / 2 + t["token"] + t["f2a"] + t["a2f"] + sw * fl
        elif isinstance(b, blocks.UpdateBlock):
            t["f2a"] = crit.cross_attn_loss(match, torch.transpose(b.f2a_attn_logit, 1, 2), dim=1)
            t["a2f"] = crit.cross_attn_loss(match, b.a2f_attn_logit, dim=2)
            t["smooth"] = loss_m.smooth_loss(b.a2f_attn_logit) + loss_m.smooth_loss(torch.transpose(b.f2a_attn_logit, 1, 2)) + fl
            w = {"frame": 1.0, "token": 1.0, "f2a": 1.0, "a2f": 1.0, "smooth": sw}
            total = t["token"] + t["f2a"] + t["a2f"] + t["frame"] + sw * t["smooth"]
        else:
            raise TypeError(type(b).__name__)
        official_total = b.compute_loss(crit, match)
        if not torch.allclose(total, official_total, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"{type(b).__name__}: decomposed {float(total)} != official {float(official_total)}")
        per_block.append({**t, "total": official_total, "weights": w})   # weights: each term's factor in the block total
    return {"blocks": per_block, "total": sum(p["total"] for p in per_block) / len(per_block),
            "match": (match[0].clone(), match[1].clone()), "n_segments": len(crit.transcript)}


def parameter_group(name: str, model: nn.Module) -> str:
    """Readable group of a parameter: the action queries, or block<i>.<type>.<part>."""
    if name == "action_query":
        return "action_query"
    parts = name.split(".")
    if parts[0] == "block_list":
        i = int(parts[1]); kind = type(model.block_list[i]).__name__
        part = {"frame_branch": "frame_branch", "action_branch": "action_branch", "f2a_layer": "f2a", "a2f_layer": "a2f",
                "seg_update": "tdu_segments", "seg_combine": "tdu_segments", "sf_merge": "tdu_merge"}.get(parts[2], parts[2])
        return f"block{i}.{kind}.{part}"
    return parts[0]
