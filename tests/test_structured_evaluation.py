import importlib
import numpy as np
import torch
from types import SimpleNamespace

def test_label_evaluation_reuses_complete_prefix_without_model_or_labels_as_input(tmp_path):
    m=importlib.import_module('echo_routing.temporal.structured_evaluation');calls=[]
    class Model:
        def decode(self,*args):
            assert len(args)==2;calls.append(len(args[0]));return args[0].argmax(1)
    p=torch.tensor([[.6,.1,.1,.1,.1],[.1,.1,.1,.1,.6]])
    bank=SimpleNamespace(sequences=[{'id':'private','labels':np.array([4,0])}],values=[(p,p,torch.tensor([4,0]))])
    pred,record=m.decode_bank(Model(),bank,tmp_path,'fixed');assert pred['private'].tolist()==[0,4] and calls==[2] and record['sequences']==1
    pred,record=m.decode_bank(Model(),bank,tmp_path,'fixed');assert calls==[2] and record['new_decodes']==0
