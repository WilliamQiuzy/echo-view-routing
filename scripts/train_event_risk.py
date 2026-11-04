#!/usr/bin/env python3
"""A62 four fixed6000 CPU event heads; no CAL/AUDIT/outer callbacks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.event_risk_common import *
from scripts.check_event_risk import tensor_data,identity
from echo_routing.temporal.event_risk_model import EventRisk,optimizer
from echo_routing.temporal.event_risk_loop import train_one
from echo_routing.temporal.event_risk_audit import FIXED_A61


def state_digest(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def main():
    p=execution_parser(__doc__);p.add_argument('--all-four-final6000',action='store_true');args=p.parse_args()
    if not authorize(args):return 0
    freeze=verify_freeze('formal_freeze.json');inputs=read(PUB/'training_inputs.json');admit=read(PUB/'formal_admission.json')
    if not args.all_four_final6000 or not admit['all_caps_fit'] or admit['training_inputs_sha256']!=sha(PUB/'training_inputs.json') or not read(PUB/'FIT_checks.json')['passed'] or (PUB/'formal_completion.json').exists():raise ValueError('complete actualFIT72/fullmatrix admission and freshfourhead run required')
    for r in FIXED_A61:verify_pin(r)
    data=np.load(verify_pin(inputs['dataset']),allow_pickle=True).item();td=tensor_data(data);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);records=[];initial={};done=0
    for seed in [2026110461,2026110462]:
        plan=np.load(verify_pin(inputs['plans'][str(seed)]),allow_pickle=True).item()
        for arm in ['R1','R2']:
            model=EventRisk(arm,seed);opt=optimizer(model);init=state_digest(model);initial.setdefault(str(seed),init)
            if initial[str(seed)]!=init:raise ValueError('matched same-seed initial tensors differ')
            ident=identity(inputs,seed,freeze);folder=RUN/'models'/f'seed{seed}'/arm
            def progress(cursor,row):write(RUN/'runtimecoordination.json',{'phase':'formal_four_event_heads6000','seed':seed,'arm':arm,'committed_updates':cursor,'actual_Adam_updates':cursor,'prior_completed_model_updates':done,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'finishall24000_thenindependentmodelaudit_CALonly6taus_AUDITdiagnostic_actualGitseal_beforeouter'})
            result=train_one(model,opt,td,plan,ident,folder,on_progress=progress);done+=result['committed_physical_updates'];records.append({'seed':seed,'arm':arm,'step':6000,'parameters':6625,'initial_state_sha256':init,'identity':ident,'path':str((folder/'step6000.pt').relative_to(ROOT)),'sha256':sha(folder/'step6000.pt'),'result':result})
    if done!=24000 or len(records)!=4 or torch.cuda.is_initialized():raise ValueError('exactfour24k admitted CPU matrix required')
    for r in FIXED_A61:verify_pin(r)
    write(PUB/'formal_completion.json',{'models':records,'physical_models':4,'formal_optimizer_updates':done,'same_seed_initial_state_sha256':initial,'matched_risk_draw_exposure':True,'CAL_AUDIT_or_outer_used_for_training_or_checkpoint_selection':False,'CUDA_initialized':False,'freeze_git':freeze['git_commit']});write(RUN/'runtimecoordination.json',{'phase':'all24000_complete_pending_models_audit_CAL_AUDIT','model_pid':os.getpid(),'formal_optimizer_updates':24000,'epoch':time.time(),'next_action':'independent16fullstatestraceaudit_thenCALthresholds_AUDIT26rows_Gitseal'});print({'formal_updates':24000,'models':4},flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
