"""A61 complete model/calibration/endpoint seal, required before outer forwards."""
import math
from echo_routing.temporal.safety_training import SEEDS,ARMS
from echo_routing.temporal.safety_git import verify as verify_Git


def validate(models,seal,model_sha,calibration_sha,plan_sha,proof=None):
    expected={(s,a) for s in SEEDS for a in ['risk',*ARMS]};utility={(s,a) for s in SEEDS for a in ARMS}
    rows=models['models'];inner=models['inner_records'];thresholds=models['thresholds']
    if not models['passed'] or models['formal_optimizer_updates']!=60000 or models['physical_models']!=10 or not models['all10finals24inner8taus56sets_frozen'] or models['inner_policy_sets']!=56:raise ValueError('complete60000/10/24/8/56 audited model barrier required')
    if len(rows)!=10 or {(r['seed'],r['arm']) for r in rows}!=expected or len(inner)!=24 or {(r['seed'],r['arm'],r['step']) for r in inner}!={(s,a,k) for s,a in utility for k in [2000,4000,6000]} or len(thresholds)!=8 or {(r['seed'],r['arm']) for r in thresholds}!=utility:raise ValueError('exact model/inner/calibration endpoint coverage required')
    if any(not math.isfinite(r['tau']) or not 0<=r['tau']<=1 for r in thresholds):raise ValueError('frozen finite8 unit-range thresholds required')
    if seal['models_manifest_sha256']!=model_sha or seal['calibration_sha256']!=calibration_sha or models['calibration_sha256']!=calibration_sha or models.get('readout_plan_sha256')!=plan_sha or seal['readout_plan_sha256']!=plan_sha:raise ValueError('actualGit preouter model/calibration/endpoint seal changed')
    final={(r['path'],r['sha256']) for r in rows}
    if len(seal['all_final_checkpoints'])!=10 or {(r['path'],r['sha256']) for r in seal['all_final_checkpoints']}!=final:raise ValueError('complete10final checkpoint SHA coverage required')
    if proof is None or 'git_commit' not in seal:raise ValueError('actual committed Git objects required, not a loose JSON seal')
    prefix='docs/results/2026-11-04_action_safety_ranking/'
    verify_Git(proof,seal['git_commit'],{prefix+'models_manifest.json':model_sha,prefix+'calibration.json':calibration_sha,prefix+'readout_plan.json':plan_sha})
    return True
