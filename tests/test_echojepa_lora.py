"""A41: pinned upstream import, strict loading, QKV LoRA, initialisation, readout and gradient paths (tiny encoder, CPU)."""
import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import echo_routing.temporal.echojepa_lora as jl

pytestmark = pytest.mark.skipif(not (jl.UPSTREAM / "src").is_dir() or not jl.PYDEPS.is_dir(), reason="pinned EchoJEPA clone / round pydeps absent")
TINY = dict(img_size=(32, 32), num_frames=4, patch_size=16, tubelet_size=2, use_rope=True, uniform_power=True, use_sdpa=True, use_activation_checkpointing=False, use_silu=False, wide_silu=True)


def _logits(m, tok: torch.Tensor) -> torch.Tensor:
    """The tiny encoder's mean tokens, zero-padded to the head's 2048 inputs."""
    f = torch.cat([tok.mean(1), tok.mean(1)], 1)
    return m.head(F.pad(f, (0, 2 * jl.DIM - f.shape[1])))


def _tiny(seed: int = 0, **kw):
    torch.manual_seed(seed)
    return jl.upstream().vit_tiny(**{**TINY, **kw}).eval()


def test_upstream_resolves_to_pinned_clone_only():
    vit = jl.upstream(); files = jl.upstream_files()
    assert Path(vit.__file__).resolve().is_relative_to(jl.UPSTREAM.resolve()) and jl.src_paths() == [(jl.UPSTREAM / "src").resolve()]
    assert "src.models.vision_transformer" in files and all(not v.startswith("/") for v in files.values()) and jl.pydeps()["in_round_dir"]


def test_constants():
    assert jl.LORA_PARAMS == 786432 and jl.HEAD_PARAMS == 10245 and jl.TRAINABLE == 796677 and jl.N_TOKENS == 1568


def test_lora_formula_and_zero_b():
    torch.manual_seed(1); base = torch.nn.Linear(16, 48); mod = jl.LoRAQKV(base); x = torch.randn(3, 5, 16)
    assert torch.equal(mod(x), base(x)) and not any(p.requires_grad for p in base.parameters())
    with torch.no_grad():
        mod.A.normal_(); mod.B.normal_()
    ref = x.double() @ base.weight.double().T + base.bias.double() + 2.0 * (x.double() @ mod.A.double().T) @ mod.B.double().T
    assert (mod(x).double() - ref).abs().max() <= 1e-5 * max(1.0, float(ref.abs().max())) and mod.scale == 2.0   # the configured relative bound


def test_wrapped_zero_lora_equals_unwrapped_and_counts():
    enc = _tiny(); off = copy.deepcopy(enc); m = jl.Recogniser(enc, bf16=False); x = torch.randint(0, 256, (2, 4, 32, 32, 3), dtype=torch.uint8)
    with torch.no_grad():
        assert torch.equal(m.tokens(x), off(jl.jepa_input(x)))
    d = enc.embed_dim; assert sum(p.numel() for p in m.parameters() if p.requires_grad) == len(enc.blocks) * jl.RANK * 4 * d + jl.HEAD_PARAMS
    assert not any(p.requires_grad for _, p in m.base_named()) and len(m.loras) == len(enc.blocks)


def test_init_identical_across_arms_and_rng_untouched():
    a, b = jl.Recogniser(_tiny(0), bf16=False), jl.Recogniser(_tiny(5), bf16=False); torch.manual_seed(123); r0 = torch.get_rng_state()
    jl.init_trainable(a, 0, 0); jl.init_trainable(b, 0, 0); assert torch.equal(torch.get_rng_state(), r0)
    sa, sb = a.trainable_state(), b.trainable_state(); assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert all(float(v.abs().sum()) == 0 for k, v in sa.items() if k.endswith(".B") or k == "head.bias")
    jl.init_trainable(b, 0, 1); assert not torch.equal(sa["lora.0.A"], b.trainable_state()["lora.0.A"])


def test_strict_loading_rules():
    src = _tiny(3); sd = {f"module.backbone.{k}": v.clone() for k, v in src.state_dict().items()}
    enc = _tiny(4); st, rep = jl.strict_encoder_state({"target_encoder": {**sd, "module.pos_embed": torch.zeros(1, 8, 192)}, "encoder": {}}, enc)
    jl.load_base(enc, st); assert all(torch.equal(enc.state_dict()[k], v) for k, v in src.state_dict().items()) and rep["dropped"][0]["key"] == "pos_embed"
    with pytest.raises(KeyError):
        jl.strict_encoder_state({"encoder": sd}, enc)
    with pytest.raises(KeyError):
        jl.strict_encoder_state({"target_encoder": {k: v for k, v in sd.items() if not k.endswith("norm.weight")}}, enc)
    with pytest.raises(KeyError):
        jl.strict_encoder_state({"target_encoder": {**sd, "module.extra": torch.zeros(1)}}, enc)
    bad = dict(sd); bad["module.backbone.norm.weight"] = torch.zeros(5)
    with pytest.raises(KeyError):
        jl.strict_encoder_state({"target_encoder": bad}, enc)


def test_readout_features_and_input():
    tok = torch.randn(3, jl.N_TOKENS, 6); mk = torch.tensor([0, 7, 15]); f = jl.readout_features(tok, mk)
    for b in range(3):
        t = int(mk[b]) // 2; assert torch.allclose(f[b], torch.cat([tok[b, t * 196:(t + 1) * 196].mean(0), tok[b].mean(0)]), atol=1e-6)
    with pytest.raises(ValueError):
        jl.readout_features(tok, torch.tensor([0, 1, 16]))
    u8 = torch.randint(0, 256, (1, 16, 8, 8, 3), dtype=torch.uint8); x = jl.jepa_input(u8)
    assert x.shape == (1, 3, 16, 8, 8) and torch.allclose(x[0, 1, 3], (u8[0, 3, :, :, 1].float() / 255 - 0.456) / 0.224, atol=1e-6)


def test_checkpointing_and_microbatch_gradients():
    enc = _tiny(2, use_activation_checkpointing=True); m = jl.Recogniser(enc, bf16=False); jl.init_trainable(m, 0, 0)
    with torch.no_grad():
        for x in m.loras:
            x.B.normal_(0, 0.02)
    x = torch.randint(0, 256, (4, 4, 32, 32, 3), dtype=torch.uint8); y = torch.tensor([0, 1, 2, 3])
    g = lambda: {k: p.grad.clone() for k, p in m.trainable()}  # noqa: E731
    F.cross_entropy(_logits(m, m.tokens(x)), y).backward(); on = g(); m.zero_grad(set_to_none=True)
    enc.use_activation_checkpointing = False; F.cross_entropy(_logits(m, m.tokens(x)), y).backward(); off = g(); m.zero_grad(set_to_none=True)
    for i in range(4):
        (F.cross_entropy(_logits(m, m.tokens(x[i:i + 1])), y[i:i + 1]) / 4).backward()
    mb = g(); assert all(float(on[k].abs().sum()) > 0 for k in on if k.endswith((".A", ".B")))
    assert max(float((on[k] - off[k]).abs().max()) for k in on) < 1e-6 and max(float((mb[k] - off[k]).abs().max()) for k in on) < 1e-5


def test_zero_b_start_gives_zero_a_gradient():
    m = jl.Recogniser(_tiny(2), bf16=False); jl.init_trainable(m, 0, 0); x = torch.randint(0, 256, (2, 4, 32, 32, 3), dtype=torch.uint8)
    F.cross_entropy(_logits(m, m.tokens(x)), torch.tensor([0, 1])).backward()
    assert all(float(l.A.grad.abs().sum()) == 0 for l in m.loras) and all(float(l.B.grad.abs().sum()) > 0 for l in m.loras)
