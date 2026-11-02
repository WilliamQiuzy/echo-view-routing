"""A57-MD official four-depth output contract; original producer stays immutable."""
import hashlib,json
from pathlib import Path
import torch
from echo_routing.temporal import jepa21_source as source
from echo_routing.temporal.dense_sequence import DenseSequence

LAYERS=(2,5,8,11)
IMAGE_ONLY=('img_mod_embed','patch_embed_img.proj.weight','patch_embed_img.proj.bias')
FEATURE_CONTRACT={'kind':'official_four_depth_concat','normalized_blocks_zero_based':list(LAYERS),'norms_in_order':['norms_block.0','norms_block.1','norms_block.2','norms_block.3'],'official_concat_axis':2,'feature_dim':3072,'spatial_mean_patches':196,'tubelets_per64_positions':32,'return_hierarchical':True,'forward_training_argument':False,'out_layers':None}

def enable_official_hierarchy(encoder):
    if (tuple(getattr(encoder,'hierarchical_layers',()))!=LAYERS or tuple(getattr(encoder,'out_layers_distillation',()))!=LAYERS or getattr(encoder,'out_layers','missing') is not None or len(getattr(encoder,'norms_block',()))!=4):raise ValueError('exact official four normalized depth map required; no guessed slice or token concatenation')
    encoder.return_hierarchical=True
    return encoder

def image_only_inactive_names(names):
    names=set(names)
    if not set(IMAGE_ONLY)<=names or not {f'norms_block.{i}.{suffix}' for i in range(4) for suffix in ['weight','bias']}<=names:raise ValueError('strict four norms and three retained image roles required')
    return list(IMAGE_ONLY)

def wrap_encoder(encoder,arm,seed):
    if arm not in ['F-MD','J-MD']:raise ValueError('predeclared matched four-depth arms required')
    enable_official_hierarchy(encoder);inactive=image_only_inactive_names(dict(encoder.named_parameters()));model=DenseSequence(encoder,'F' if arm=='F-MD' else 'J-high',dim=3072,channels=128,bf16=True,seed=seed,inactive_names=inactive);model.feature_contract={**FEATURE_CONTRACT,'normalized_blocks_zero_based':list(LAYERS),'norms_in_order':list(FEATURE_CONTRACT['norms_in_order'])};model.external_arm=arm
    return model

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def build_model(cfg,arm,seed,device='cuda'):
    # Definition only during preparation. Real forwards are a separately admitted stage.
    manifest=json.loads((source.ROOT/cfg['source_manifest']).read_text());path=source.ROOT/cfg['source_state'];assert manifest['passed'] and sha(path)==manifest['target_encoder_sha256']
    for name,digest in manifest['loaded_model_subset_SHA'].items():assert sha(source.UPSTREAM/name)==digest,name
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    encoder=source.build_encoder();state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);expected=encoder.state_dict();assert len(expected)==len(state)==158 and set(expected)==set(state) and all(expected[k].shape==state[k].shape for k in state);encoder.load_state_dict(state,strict=True);assert all(torch.equal(encoder.state_dict()[k],v) for k,v in state.items());model=wrap_encoder(encoder,arm,seed)
    assert sum(p.numel() for p in model.encoder.parameters())==86833152 and sum(p.numel() for p in model.head.parameters())==1184906
    if arm=='J-MD':assert sum(p.requires_grad for p in model.encoder.parameters())==155 and sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)==86241792
    return model.to(device)
