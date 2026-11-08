import pytest


class FakeChild:
    pid=999
    def poll(self):return 0
    def wait(self,timeout=None):return 0


class State:
    def __init__(self,storage):self.values=iter(storage);self.events=[];self.records=[];self.owner={'deadline_epoch':10**20}
    def authorize(self):self.events.append('auth')
    def queue(self):
        from contextlib import nullcontext
        return nullcontext(4)
    def inventory(self):return {'GPU_owners':[],'boot_id':'boot','guardian_start_ticks':1}
    def storage(self):return next(self.values)
    def admission(self):return {'all_caps_fit':True,'stage_CPU_limit':60.,'stage_wall_limit':60.,'actual_storage_cap':1000,'budget_spent':0}
    def claim(self,stage,nonce,live):self.events.append('durable-claim');return {'path':'intent','sha256':'a'*64}
    def command(self,*a):return ['fake-child']
    def logs(self,*a):
        from contextlib import nullcontext
        return nullcontext((None,None))
    def record(self,row,intent):self.records.append(row);return row
    def usage(self):return 0.


def test_storage_overcap_rejects_before_child(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_guardian import execute
    s=State([1001]);monkeypatch.setattr('subprocess.Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must never spawn')))
    row=execute(s,'observe_geometry')
    assert row['model_pid'] is None and row['termination_reason']=='prelaunch_refused'
    assert row['observed_worker_exit_code'] is None and row['stage_exit_status']==1


def test_fast_child_final_storage_is_still_a_failure(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_guardian import execute
    s=State([10,1001]);monkeypatch.setattr('subprocess.Popen',lambda *a,**k:FakeChild())
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_guardian.start_ticks',lambda p:1)
    row=execute(s,'observe_geometry')
    assert row['observed_worker_exit_code']==0 and row['actual_child_directly_waited'] is True
    assert row['stage_exit_status']==1 and row['termination_reason']=='actual_final_cap_exceeded'


def test_durable_claim_precedes_every_popen_and_wait_is_unconditional(monkeypatch):
    from echo_routing.temporal.selected_risk_resume_guardian import execute
    s=State([10,10])
    def popen(*a,**k):assert s.events[-1]=='durable-claim';s.events.append('popen');return FakeChild()
    monkeypatch.setattr('subprocess.Popen',popen);monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_guardian.start_ticks',lambda p:1)
    assert execute(s,'observe_geometry')['stage_exit_status']==0


def test_unknown_stage_never_opens_logs_or_queue():
    from echo_routing.temporal.selected_risk_resume_guardian import execute
    s=State([])
    with pytest.raises(ValueError):execute(s,'../old/file')
    assert s.events==[]
