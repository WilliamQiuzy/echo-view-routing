from pathlib import Path
import json,subprocess
import pytest

def test_stage_freeze_requires_actual_committed_bytes_and_blocks_old_cli(tmp_path,monkeypatch):
 import scripts.freeze_event_risk as m
 from echo_routing.temporal.anchor_cache import atomic_json
 root=tmp_path/'toy';root.mkdir();subprocess.run(['git','init','-q',str(root)],check=True)
 for k,v in [('user.name','Toy'),('user.email','toy@example.invalid')]:subprocess.run(['git','-C',str(root),'config',k,v],check=True)
 (root/'scripts').mkdir();(root/'scripts/new_event_risk.py').write_text('value=1\n');cfg=root/'config.json';cfg.write_text(json.dumps({k:{'path':k+'.json'} for k in ['card','authorization','basis']}))
 for k in ['card','authorization','basis']:(root/(k+'.json')).write_text('{}')
 subprocess.run(['git','-C',str(root),'add','.'],check=True);subprocess.run(['git','-C',str(root),'commit','-qm','toy'],check=True)
 run=root/'newrun';pub=root/'newpub';monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'RUN',run);monkeypatch.setattr(m,'PUB',pub);monkeypatch.setattr(m,'CFG',cfg);monkeypatch.setattr(m,'write',atomic_json);monkeypatch.setattr(m,'pin',lambda p:{'path':str(Path(p).relative_to(root)),'sha256':m.sha(p)})
 c=m.make('toy_freeze.json',['scripts/new_event_risk.py'],[]);assert len(c['objects'])==5 and (run/'toy_freeze.json').exists()
 (root/'scripts/new_event_risk.py').write_text('value=2\n')
 with pytest.raises(ValueError,match='commit every'):m.make('changed_freeze.json',['scripts/new_event_risk.py'],[])
 (root/'scripts/new_event_risk.py').write_text('from scripts.readout_action_safety import main\n')
 with pytest.raises(ValueError,match='old writing CLI'):m.source_closure(['scripts/new_event_risk.py'])
