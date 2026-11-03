import importlib
import numpy as np
import torch
from echo_routing.vit import target_mvit as tm
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.temporal.structured_inputs import windows

class Pixels:
    def frame(self,video,index,gamma):return np.full((224,224,3),index,np.uint8)

class A08:
    def __init__(self):self.calls=[]
    def __call__(self,rgb,marker):
        self.calls.append((rgb.clone(),marker.clone()));z=torch.zeros(len(rgb),5);z[:,0]=rgb[:,0].mean((1,2,3));return z

def test_A08_dense_producer_keeps_original_windows_markers_RGB_and_nearest_center():
    m=importlib.import_module('scripts.cache_structured_evidence');q={'refs':[('v',i,1.) for i in range(19)]};model=A08();p=m.a08_sequence(model,q,Pixels(),device='cpu',batch=32);w=windows(19,2)
    want=np.stack([[Pixels().frame('v',int(i),1.) for i in s] for s in w['samples']]);torch.testing.assert_close(model.calls[0][0],tm.to_rgb(torch.from_numpy(want)));assert torch.equal(model.calls[0][1],tm.marker_planes(torch.from_numpy(w['marker'])))
    z=torch.zeros(len(w['centre']),5);z[:,0]=model.calls[0][0][:,0].mean((1,2,3));expected=torch.softmax(z,1).numpy()[nearest_centre(19,w['centre'])];np.testing.assert_array_equal(p,expected)

class JEPA:
    def probabilities(self,u8,mask):
        z=torch.zeros(len(u8),64,5);z[:,:,0]=u8[:,:,:,0,0].float().mean(2)/100;return z.softmax(-1)

def test_JEPA_original_chunk_padding_and_positive_stitch_without_labels():
    m=importlib.import_module('scripts.cache_structured_evidence');q={'refs':[('v',i,1.) for i in range(73)]};p=m.jepa_sequence(JEPA(),q,Pixels(),device='cpu');assert p.shape==(73,5) and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-6);assert p[-1,0]>p[0,0]
