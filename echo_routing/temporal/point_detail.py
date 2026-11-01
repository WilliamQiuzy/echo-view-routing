"""A52 point-only direct-view spatial learning; no anatomy or deployment gate."""
import numpy as np
from PIL import Image

G_AUC=0.8074195308237862
G_FPR=4/94


def point_canvas(rgb,arm):
    x=np.asarray(rgb)
    if x.dtype!=np.uint8 or x.ndim!=3 or x.shape[2]!=3 or min(x.shape[:2])<=0 or arm not in ('H','R'):
        raise ValueError('uint8 HWC RGB and fixed H/R arm required')
    h,w=x.shape[:2];scale=448/max(w,h);nw,nh=round(w*scale),round(h*scale)
    im=Image.fromarray(x).resize((nw,nh),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(448,448),(0,0,0));canvas.paste(im,((448-nw)//2,(448-nh)//2))
    if arm=='R':canvas=canvas.resize((224,224),Image.Resampling.BILINEAR).resize((448,448),Image.Resampling.BILINEAR)
    return np.asarray(canvas).copy()


def normalise(rgb):
    import torch
    if rgb.dtype!=torch.uint8 or rgb.ndim!=4 or tuple(rgb.shape[1:])!=(448,448,3):
        raise ValueError('NHWC448 uint8 required')
    x=rgb.permute(0,3,1,2).float()/255
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


def build(state):
    import torch
    from echo_routing.temporal.pw_pilot import make_model
    m=make_model(state)
    for p in m.parameters():p.requires_grad_(True)
    with torch.no_grad():m.head.weight.zero_();m.head.bias.fill_(-2)
    return m.eval()


def source_gates(metrics,geometry_ba,constant_ba):
    checks={}
    for arm in ('H','R'):
        a=metrics[arm]
        checks[arm]={'BA':a['balanced_accuracy']>=.8,'A4C_recall':a['A4C_recall']>=.7,
          'A5C_recall':a['A5C_recall']>=.7,'A4C_protection':1-a['A4C_recall']<=G_FPR+.02+1e-12,
          'AUC_floor':a['AUROC']>=.85,'AUC_increment':a['AUROC']>=G_AUC+.03,
          'image_over_geometry':a['balanced_accuracy']>=geometry_ba+.1,
          'image_over_constant':a['balanced_accuracy']>=constant_ba+.1}
    h,r=metrics['H'],metrics['R']
    detail={'BA_increment':h['balanced_accuracy']>=r['balanced_accuracy']+.05,
      'no_extra_A4C_harm':h['A4C_recall']>=r['A4C_recall'], 'AUC_increment':h['AUROC']>=r['AUROC']+.03}
    return {'absolute_checks':checks,'absolute_pass':any(all(v.values()) for v in checks.values()),
      'detail_checks':detail,'detail_increment_pass':all(detail.values())}
