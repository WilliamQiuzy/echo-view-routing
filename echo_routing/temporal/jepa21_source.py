"""Strict official Meta2.1 model interface for the named EchoJEPA-B checkpoint."""
import importlib,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'third_party/VJEPA2_1'
COMMIT='204698b45b3712590f06245fbfba32d3be539812'
SOURCE_SHA='40384a17d3d32375142e3bd3a43d2f5bfa2b174b44c6960958bffd731e9624c1'
KW=dict(img_size=(224,224),num_frames=16,patch_size=16,tubelet_size=2,use_rope=True,uniform_power=False,use_silu=False,wide_silu=True,use_sdpa=True,use_activation_checkpointing=True,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,img_temporal_dim_size=1,interpolate_rope=True,modality_embedding=True,n_output_distillation=4)

def upstream():
    for name in ('src','app'):
        m=sys.modules.get(name)
        if m is not None:
            paths=list(getattr(m,'__path__',[]))
            if not paths or any(not Path(p).resolve().is_relative_to(UPSTREAM.resolve()) for p in paths):raise ImportError(f'foreign {name} package already loaded')
    if str(UPSTREAM) not in sys.path:sys.path.insert(0,str(UPSTREAM))
    deps=ROOT/'third_party/a41_pydeps'
    if str(deps) not in sys.path:sys.path.append(str(deps))  # Reuse the existing pinned einops0.8.1; shared .venv unchanged.
    return importlib.import_module('app.vjepa_2_1.models.vision_transformer')

def build_encoder():return upstream().vit_base(**KW)

def strict_target_state(checkpoint,model):
    raw=checkpoint['target_encoder'];state={}
    for key,value in raw.items():
        if not key.startswith('module.backbone.'):raise ValueError('unexpected target_encoder prefix')
        name=key[len('module.backbone.'):]
        if name in state or not isinstance(value,torch.Tensor):raise ValueError('duplicate/non-tensor source key')
        state[name]=value.detach().float()
    expected=model.state_dict()
    if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape for k in expected):raise ValueError('strict source key/shape mismatch; no drop/fill permitted')
    model.load_state_dict(state,strict=True)
    if not all(torch.equal(model.state_dict()[k],v) for k,v in state.items()):raise ValueError('loaded state not exact')
    return state

def inactive_video_names(names):
    return [n for n in names if n=='img_mod_embed' or n.startswith('patch_embed_img.') or n.startswith(('norms_block.0.','norms_block.1.','norms_block.2.'))]
