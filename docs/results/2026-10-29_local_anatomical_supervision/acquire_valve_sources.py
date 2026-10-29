"""No-forward inspection of exactly licensed, pinned localization weights."""
from pathlib import Path
import hashlib
import inspect
import json
import time
import sys
sys.path.insert(0,'/home/william/echo-view-routing')
import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from echo_routing.temporal.valve_localized import canonical_backbone

ROOT=Path('/home/william/echo-view-routing/runs/a48_valve_source_intake_v1')
RURL='https://huggingface.co/CardiovascularImagingVHIR/AoVdetector/resolve/d35108887c9bcd432d852da103e684db6c2fd969/model_avdetector.pth'
RSHA='20951944cad682ea8dfa8ca9635ac12411123453b6f4b9c6702fa958974b8557'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    w=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    expected=resnet_fpn_backbone(backbone_name='resnet50',weights=None,norm_layer=FrozenBatchNorm2d,trainable_layers=3).state_dict()
    sources={};states={}
    for arm,url in [('R',RURL),('G',w.url)]:
        path=ROOT/(('model_avdetector.pth') if arm=='R' else Path(url).name)
        if not path.exists():torch.hub.download_url_to_file(url,str(path),progress=False)
        if arm=='R':assert path.stat().st_size==165770983 and sha(path)==RSHA
        assert path.stat().st_size<200_000_000
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
        classes=4 if arm=='R' else 91
        assert tuple(state['roi_heads.box_predictor.cls_score.weight'].shape)==(classes,1024)
        assert tuple(state['roi_heads.box_predictor.bbox_pred.weight'].shape)==(4*classes,1024)
        selected,renamed=canonical_backbone(state,expected)
        assert all(v.shape==expected[k].shape and v.dtype==expected[k].dtype and torch.isfinite(v).all() for k,v in selected.items())
        torch.save(selected,ROOT/f'backbone_{arm}.pt');states[arm]=selected
        sources[arm]={'url':url,'original_bytes':path.stat().st_size,'original_sha256':sha(path),'backbone_bytes':(ROOT/f'backbone_{arm}.pt').stat().st_size,'backbone_sha256':sha(ROOT/f'backbone_{arm}.pt'),'backbone_keys':len(selected),'classes_unused':classes,'strict_complete_load':True,'safe_weights_only':True,'documented_FPN_v1_v2_renames':renamed}
    changed=[k for k in expected if not torch.equal(states['R'][k],states['G'][k])]
    assert len(changed)>=40 and any(k.startswith('body.layer4.') for k in changed) and any(k.startswith('fpn.') for k in changed),'source localization backbone did not genuinely change'
    # FrozenBN buffers are intentionally constant; do not count them as trainable evidence.
    result={'sources':sources,'backbone_changed_R_vs_G':len(changed),'backbone_state_tensors':len(expected),'torch':torch.__version__,'torchvision':torchvision.__version__,'torchvision_backbone_source_sha256':sha(inspect.getfile(resnet_fpn_backbone)),'FrozenBatchNorm_eps_required':0.,'license_R':'CC-BY-NC-SA-4.0 weights; upstream repository code not reused','model_forward_count':0,'gpu_hours':0,'written_unix':time.time()}
    (ROOT/'sources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=1))


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);st=time.time();cpu=time.process_time();ok=False
    try:run();ok=True
    finally:(ROOT/'receipt.json').write_text(json.dumps({'ok':ok,'wall_seconds':time.time()-st,'process_cpu_seconds':time.process_time()-cpu,'GPU_hours':0,'script_sha256':sha(__file__),'finished_unix':time.time()},indent=2)+'\n')
