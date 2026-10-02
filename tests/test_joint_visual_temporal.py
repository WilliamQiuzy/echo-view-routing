"""Joint fine-tuning checks (round 2, Task 5b): the chunked, checkpointed encoder reproduces whole-batch gradients
(fp32, dropout off), frozen weights get no gradient, preprocessing equals the E1 extraction, padding equals collate."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from echo_routing.temporal.dev_data import Seq, collate
from echo_routing.temporal.dev_heads import BoundaryContextHead
from echo_routing.temporal.joint_visual import VisualEncoder, crop_uint8, load_frame_u8, trainable_blocks_from
from scripts.train_joint_visual_temporal import padded


class FakeBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__(); self.ln = nn.LayerNorm(d); self.fc = nn.Linear(d, d)

    def forward(self, x):
        return x + self.fc(self.ln(x))


class FakeViT(nn.Module):
    """Same attributes and forward order as the official CLIP VisionTransformer, tiny sizes."""

    def __init__(self, d=16, patch=8, res=16, layers=3, out=8):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Conv2d(3, d, patch, patch, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(d) * 0.1)
        self.positional_embedding = nn.Parameter(torch.randn((res // patch) ** 2 + 1, d) * 0.1)
        self.ln_pre = nn.LayerNorm(d)
        self.transformer = nn.Module(); self.transformer.resblocks = nn.Sequential(*[FakeBlock(d) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(d); self.proj = nn.Parameter(torch.randn(d, out) * 0.1)

    def forward(self, x):
        x = self.conv1(x); x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]), x], 1) + self.positional_embedding
        x = self.ln_pre(x).permute(1, 0, 2); x = self.transformer.resblocks(x).permute(1, 0, 2)
        return self.ln_post(x[:, 0, :]) @ self.proj


def _u8(n=5, res=16, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, size=(n, res, res, 3), dtype=np.uint8))


def test_encoder_matches_the_official_forward_and_freezes_the_prefix():
    vt = FakeViT(); enc = VisualEncoder(vt, trainable_blocks=1, chunk=2).eval()
    u8 = _u8()
    with torch.no_grad():
        assert torch.allclose(enc(u8), vt(enc.normalise(u8)), atol=1e-6)
    names = {n for n, _ in enc.trainable()}
    assert names == {"transformer.resblocks.2.ln.weight", "transformer.resblocks.2.ln.bias", "transformer.resblocks.2.fc.weight",
                     "transformer.resblocks.2.fc.bias", "ln_post.weight", "ln_post.bias", "proj"}


def test_chunked_checkpointed_gradients_equal_whole_batch_gradients():
    a = VisualEncoder(FakeViT(), 1, chunk=2).train(); b = VisualEncoder(copy.deepcopy(a.vt), 1, chunk=1000).train()
    u8 = _u8(7)
    (a(u8) ** 2).sum().backward(); (b(u8) ** 2).sum().backward()
    for (na, pa), (nb, pb) in zip(a.trainable(), b.trainable()):
        assert na == nb and torch.allclose(pa.grad, pb.grad, atol=1e-5)
    assert all(p.grad is None for n, p in a.vt.named_parameters() if not p.requires_grad)


def _joint_grads(chunk: int, seqs, head, head_model, vt):
    enc = VisualEncoder(copy.deepcopy(vt), 1, chunk=chunk).train(); hm = copy.deepcopy(head_model).train()
    counts = [s.labels.size for s in seqs]; u8 = _u8(sum(counts), seed=3)
    batch = collate(seqs); batch["x"] = padded(enc(u8), counts)
    loss, _ = head.loss(hm, batch); loss.backward()
    return float(loss), {n: p.grad.clone() for n, p in enc.trainable()}, {n: p.grad.clone() for n, p in hm.named_parameters() if p.grad is not None}


def test_joint_loss_gradients_do_not_depend_on_chunking_for_the_pooled_model():
    seqs = [Seq("st", False, np.zeros((9, 1), np.float16), np.array([0] * 5 + [2] * 4), np.array([5])),
            Seq("na@0", True, np.zeros((6, 1), np.float16), np.full(6, 1), np.zeros(0, int))]
    head = BoundaryContextHead({"in_dim": 8, "dim": 16, "num_classes": 5, "heads": 4, "radius": 3, "dropout": 0.0,
                                "use_path": True, "segment_pool": True}, 0.2, "full_pool")
    torch.manual_seed(1); hm = head.build()
    la, ga, ha = _joint_grads(2, seqs, head, hm, FakeViT()); lb, gb, hb = _joint_grads(1000, seqs, head, hm, FakeViT())
    assert abs(la - lb) < 1e-6
    assert all(torch.allclose(ga[k], gb[k], atol=1e-5) for k in ga) and all(torch.allclose(ha[k], hb[k], atol=1e-5) for k in ha)


def test_padding_matches_collate():
    f = torch.arange(15, dtype=torch.float32).view(5, 3)
    x = padded(f, [2, 3])
    assert x.shape == (2, 3, 3) and torch.equal(x[0, :, :2], f[:2].T) and torch.equal(x[0, :, 2], torch.zeros(3)) and torch.equal(x[1], f[2:].T)


def test_uint8_preprocessing_equals_the_e1_float_pipeline(tmp_path):
    from scripts.run_windowed_echoviewclip import apply_gamma, preprocess
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)); path = tmp_path / "f.jpg"; im.save(path)
    enc = VisualEncoder(FakeViT(res=224, patch=16), 1)
    for gamma in (1.0, 0.9):
        with Image.open(path) as j:
            ref = preprocess(apply_gamma(j.convert("RGB"), gamma))
        got = enc.normalise(torch.from_numpy(load_frame_u8(path, gamma))[None])[0]
        assert torch.allclose(got, ref, atol=1e-5)
    assert crop_uint8(im).shape == (224, 224, 3)


REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "cache_v2" / "e1_raw" / "a74c7b233e79" / "visual_encoder_fp32.pt"


@pytest.mark.skipif(not EXPORT.exists() or not (REPO / "third_party" / "EchoViewCLIP" / "clip").exists(), reason="server-only")
def test_official_encoder_reproduces_the_exported_reference_embeddings():
    from echo_routing.temporal.joint_visual import load_vision_transformer
    vt, _ = load_vision_transformer(REPO / "third_party" / "EchoViewCLIP", EXPORT)
    enc = VisualEncoder(vt, 1).eval(); ref = json.loads(EXPORT.with_suffix(".json").read_text())
    u8 = torch.from_numpy(np.stack([load_frame_u8(Path(p), 1.0) for p in ref["reference_frames"]]))
    with torch.no_grad():
        assert float((enc(u8) - torch.tensor(ref["reference_embeddings"])).abs().max()) < 1e-3


@pytest.mark.skipif(not (REPO / "third_party" / "asrf" / "libs").exists(), reason="official ASRF checkout is server-only")
def test_joint_loss_gradients_do_not_depend_on_chunking_for_asrf():
    from echo_routing.temporal.dev_heads import ASRFHead, asrf_weights
    seqs = [Seq("st", False, np.zeros((9, 1), np.float16), np.array([0] * 5 + [2] * 4), np.array([5])),
            Seq("na@0", True, np.zeros((6, 1), np.float16), np.array([1, 1, 3, 3, 4, 4]), np.zeros(0, int))]
    cw, pw = asrf_weights(seqs, 5)
    head = ASRFHead(REPO / "third_party" / "asrf", 8, cw, pw)
    torch.manual_seed(2); hm = head.build()
    for m in hm.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    la, ga, ha = _joint_grads(2, seqs, head, hm, FakeViT()); lb, gb, hb = _joint_grads(1000, seqs, head, hm, FakeViT())
    assert abs(la - lb) < 1e-5
    assert all(torch.allclose(ga[k], gb[k], atol=1e-5) for k in ga) and all(torch.allclose(ha[k], hb[k], atol=1e-5) for k in ha)


def test_unfreeze_list_must_name_the_last_blocks_ln_post_and_proj():
    cfg = ["image_encoder.transformer.resblocks.11", "image_encoder.ln_post", "image_encoder.proj"]
    assert trainable_blocks_from(cfg, 12) == 1
    assert trainable_blocks_from(["image_encoder.transformer.resblocks.10", *cfg], 12) == 2
    with pytest.raises(SystemExit):
        trainable_blocks_from(["image_encoder.transformer.resblocks.10", "image_encoder.ln_post", "image_encoder.proj"], 12)
    with pytest.raises(SystemExit):
        trainable_blocks_from(["image_encoder.transformer.resblocks.11", "image_encoder.ln_post"], 12)
