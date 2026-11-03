import importlib,json
import numpy as np
import pytest,torch
from echo_routing.temporal.structured_evidence import StructuredEvidence
from echo_routing.temporal.structured_training import optimizer

def test_journal_latest_cursor_and_immutable_milestones_resume_without_optimizer_replay(tmp_path):
    m=importlib.import_module('echo_routing.temporal.structured_loop');batch=[(torch.full((n,5),.2),torch.full((n,5),.2),torch.arange(n)%5) for n in [9,11]];plan=np.tile([0,1],(4,1));called=[]
    def run(stop):
        h=StructuredEvidence('D2',58);o=optimizer(h);m.run_updates(h,o,lambda _:batch,plan,tmp_path,'fixed',stop=stop,checkpoint_every=2,milestones=[2,4],on_milestone=lambda step,path:called.append(step));return h,o
    run(2);h,o=run(4);rows=[json.loads(s) for s in (tmp_path/'physical_updates.jsonl').read_text().splitlines()];assert [r['u'] for r in rows]==[0,1,2,3] and called==[2,2,4];assert (tmp_path/'step2.pt').exists() and (tmp_path/'step4.pt').exists();assert all(int(s['step'])==4 for s in o.state.values())
    z=torch.load(tmp_path/'step4.pt',weights_only=True);assert all(torch.equal(z['model'][k],v) for k,v in h.state_dict().items())
    with (tmp_path/'physical_updates.jsonl').open('a') as f:f.write(json.dumps({**rows[-1],'u':4})+'\n')
    with pytest.raises(RuntimeError):run(4)

def test_ambiguous_pending_training_update_stops_without_any_new_step(tmp_path):
    m=importlib.import_module('echo_routing.temporal.structured_loop');(tmp_path/'pending.json').write_text('{}');h=StructuredEvidence('D2',58);o=optimizer(h)
    with pytest.raises(RuntimeError):m.run_updates(h,o,lambda _:None,np.tile([0,1],(4,1)),tmp_path,'fixed',stop=4)
    assert not o.state
