"""PW-T1 real image training and complete shared-reference model bundles."""
from __future__ import annotations
import hashlib
import torch
import torch.nn.functional as F
from echo_routing.temporal.pw_pilot import make_model

BASE_SHA='4ae0006ff349897f5b73d9c20da9ff33919f2d23db2cc1b7a5e00e9c08b1d42a'
SOURCE_SHA={'L':'687ca2adae20eccc08b304c987985d209295f9b73b3e4e24de8eb65f93395762',
            'S':'dd08954ed78c99936486abbbea3a320f09d447e613eedd3aa567d69e912192e2','G':BASE_SHA}


def state_sha(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def normalise(frames):
    if frames.dtype!=torch.uint8 or frames.ndim!=4 or frames.shape[-1]!=3:raise ValueError('expectedNHWCuint8')
    x=frames.permute(0,3,1,2).float()/255
    mean=torch.tensor([.485,.456,.406],device=x.device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=x.device).view(1,3,1,1)
    return (x-mean)/std


def build(base_state,source_state=None):
    m=make_model(base_state)
    live={n for n,p in m.named_parameters() if p.requires_grad}
    frozen={n:t.detach().clone() for n,t in m.state_dict().items() if n not in live}
    if source_state is not None:m.load_state_dict(source_state,strict=True)
    with torch.no_grad():m.head.weight.zero_();m.head.bias.fill_(-2)
    assert all(torch.equal(m.state_dict()[n],v) for n,v in frozen.items()),'source nontrainable tensors differ from shared G'
    return m.eval()


def make_bundle(model,base_sha,metadata):
    live={n for n,p in model.named_parameters() if p.requires_grad}
    state=model.state_dict()
    return {'base_sha256':base_sha,'trainable':{n:state[n].detach().cpu().clone() for n in sorted(live)},
            'full_state_sha256':state_sha(state),'frozen_state_sha256':state_sha({n:t for n,t in state.items() if n not in live}),
            'metadata':metadata}


def restore_bundle(model,bundle,base_sha):
    if bundle['base_sha256']!=base_sha:raise ValueError('wrong complete-model reference')
    live={n for n,p in model.named_parameters() if p.requires_grad}
    if set(bundle['trainable'])!=live:raise ValueError('selected tensors incomplete')
    frozen={n:t for n,t in model.state_dict().items() if n not in live}
    if state_sha(frozen)!=bundle['frozen_state_sha256']:raise ValueError('shared frozen state mismatch')
    missing,unexpected=model.load_state_dict(bundle['trainable'],strict=False)
    if unexpected or set(missing)!=set(frozen):raise ValueError('model reconstruction incomplete')
    if state_sha(model.state_dict())!=bundle['full_state_sha256']:raise ValueError('full model state mismatch')
    return model.eval()


def leaf_coefficients(scores,weights,segments):
    leaf=scores.detach().clone().requires_grad_(True)
    weights=weights.to(device=leaf.device,dtype=leaf.dtype);segments=segments.to(leaf.device)
    r=torch.stack([(leaf[segments==k]*weights[segments==k]).sum() for k in range(4)])
    loss=F.softplus(-r[0])+F.softplus(r[1:]).mean()
    g=torch.autograd.grad(loss,leaf)[0]
    return loss.detach(),r.detach(),g.detach()


def two_pass(model,frames,rows,weights,segments,device):
    with torch.no_grad():
        scores=torch.cat([model(normalise(torch.from_numpy(frames[int(r)][None]).to(device))) for r in rows])
    loss,r,coeff=leaf_coefficients(scores,weights,segments)
    replay=0.
    for k,row in enumerate(rows):
        value=model(normalise(torch.from_numpy(frames[int(row)][None]).to(device))).squeeze(0)
        replay=max(replay,float(abs(value.detach()-scores[k]).item()))
        value.backward(coeff[k])
    return {'loss':float(loss),'run_scores':r.cpu().tolist(),'replay_max_abs':replay,
            'occurrences':len(rows),'physical_forwards':2*len(rows),'physical_backwards':len(rows)}
