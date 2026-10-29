"""Derive explicit head artifacts from the four completed full checkpoints.
No model instantiation, forward, optimizer or heldout read. Sole-owner metadata.
"""
import hashlib,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
import scripts.train_valve_localized_a5c as a

def run():
    p=a.OUT/'weights_s0.json';m=json.loads(p.read_text());assert len(m['models'])==4
    for key,v in m['models'].items():
        rd=a.OUT/'train'/key;assert a.sha(rd/'final.pt')==v['final_sha256']
        full=torch.load(rd/'final.pt',map_location='cpu',weights_only=True)
        head={k.removeprefix('head.'):z for k,z in full.items() if k.startswith('head.')}
        assert sum(z.numel() for z in head.values())==294281
        hp=rd/'head.pt';assert not hp.exists(),'head already derived; verify existing metadata rather than regenerate'
        torch.save(head,hp)
        v.update(head_sha256=a.sha(hp),head_bytes=hp.stat().st_size,head_tensor_sha256={k:a.digest(z.numpy()) for k,z in head.items()})
        a.write(rd/'manifest.json',v)
    m['manifest_finalization_code_sha256']={str(Path(__file__).relative_to(ROOT)):a.sha(Path(__file__))}
    m['heads_derived_after_training_without_forward']=True;m['head_manifests_written']=a.t.now()
    a.publish('weights_s0.json',m)
    print(json.dumps({'derived_heads':4,'total_head_parameters':294281*4}))
if __name__=='__main__':
    start=time.process_time();at=a.t.now();ok=False
    try:run();ok=True
    finally:a.t.ledger_append(a.OUT/'ledger.jsonl','A48 explicit head manifest CPU','cpu',at,time.process_time()-start,ok,'no GPU, forward or heldout read; import CPU included in root allowance')
