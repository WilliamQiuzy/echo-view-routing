"""Post-A44 observational diagnosis from existing private caches; no model forward."""
import sys, time, json, math, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import scripts.train_risk_constrained_a5c as t

def stats(x):
    x=np.asarray(x,dtype=float)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,'q':{str(q):float(np.quantile(x,q)) for q in [0,.1,.5,.9,1]} if len(x) else {}}

def auc(p,n):
    p=np.asarray(p)[:,None];n=np.asarray(n)[None,:]
    return float(np.mean((p>n)+.5*(p==n)))

def main():
    started=t.now();start=time.time(); root=ROOT/'runs/video_risk_constrained_a5c_v1'
    ctx=t.Ctx(t.load_cfg(SimpleNamespace(config='configs/experiments/video_risk_constrained_a5c_v1.yaml',set=[])),'configs/experiments/video_risk_constrained_a5c_v1.yaml')
    cache=np.load(root/'readout_cache_s0.npz',allow_pickle=False)
    result={'contract_commit':'1037684','kind':'exploratory existing-output diagnosis; no efficacy rerun','fit':{},'heldout':{}}
    for f in [0,1]:
        F=t.Fold(ctx,0,f);pk=F.pk;neg=F.L['negatives'];pos=F.L['positives'];plan=F.pl
        for arm in ['C','N']:
            rd=root/'train'/f'{arm}_s0_f{f}';us=[json.loads(x) for x in (rd/'updates.jsonl').read_text().splitlines()];z=np.load(rd/'full_fit_scores.npz',allow_pickle=False)
            smap=dict(zip(z['rows'].tolist(),z['update_600'].tolist()));rf={}
            for i in F.L['eligible']:
                a,b=pk['offsets'][i:i+2];rf[int(i)]=math.fsum(float(w)*smap[int(j-F.off)] for j,w in zip(pk['rows'][a:b],pk['weights'][a:b]))
            last={};lam=1.;de=0.
            for u,rec in enumerate(us):
                assert rec['u']==u
                for i,r in zip(plan[u,1:],rec['r_neg']):last[int(i)]=(u,float(r))
                assert abs(lam-rec['lambda_before'])<1e-12
                nxt=min(100,max(0,lam+.1*(rec['H']-.01))) if arm=='N' else lam
                de=max(de,abs(nxt-rec['lambda_after']));lam=nxt
            assert set(last)==set(neg.tolist()) and de<1e-12
            broken=[int(i) for i in neg if rf[int(i)]>0];viol=[int(i) for i in neg if rf[int(i)]>-1]
            blocks={}
            for a,b in [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600),(550,600)]:
                uu=us[a:b];nocc=sum(u['n_neg_occ'] for u in uu)
                blocks[f'{a}:{b}']={'H_mean':float(np.mean([u['H'] for u in uu])),'all_three_inactive_updates':sum(u['H']==0 for u in uu),'negative_zero_grad_window_fraction':sum(u['zero_grad_neg_windows'] for u in uu)/nocc,'negative_to_positive_score_gradient_mass':sum(u['gsum_neg'] for u in uu)/sum(u['gsum_pos'] for u in uu),'lambda_range':[min(u['lambda_before'] for u in uu),max(u['lambda_after'] for u in uu)],'clip_active_fraction':sum(u['grad_norm']>1 for u in uu)/len(uu)}
            vals=np.array([rf[int(i)] for i in neg]);v= {'actual_updates':len(us),'plan_sha256':F.psha,'dual_recurrence_error':de,'blocks':blocks,'full_fit_H':float(np.maximum(0,1+vals).mean()),'fit_false_promotions':len(broken),'fit_S1_fixed':sum(rf[int(i)]>0 for i in pos),'fit_auc_A5C_vs_correct_A4C':auc([rf[int(i)] for i in pos],vals),'negative_scores':stats(vals),'positive_scores':stats([rf[int(i)] for i in pos]),'last_seen_safe_le_minus1_then_final_positive':sum(last[i][1]<=-1 for i in broken),'last_seen_nonpositive_then_final_positive':sum(last[i][1]<=0 for i in broken),'last_seen_safe_then_final_hinge_violation':sum(last[i][1]<=-1 for i in viol),'last_seen_scores_of_final_promotions':stats([last[i][1] for i in broken]),'last_encounter_to_end_updates_final_promotions':stats([599-last[i][0] for i in broken]),'all_negative_last_encounter_lag':stats([599-last[int(i)][0] for i in neg])}
            result['fit'][f'{arm}_f{f}']=v
        h=f; D=t.d32.fold_data(ctx.d,0,h);pk=D['pk'];y,b=pk['labels'],pk['base'];el=cache[f'eligible_runs_heldout_f{h}'];p=(y[el]==3);n=(y[el]==2);rc=cache[f'r_C_heldout_f{h}'];rn=cache[f'r_N_heldout_f{h}'];extra=(rn>0)&(rc<=0)
        def group(mask):
            ids=el[mask];return {'runs':len(ids),'pure':int((pk['purity'][ids]>=1).sum()),'mixed':int((pk['purity'][ids]<1).sum()),'native':sum(D['seqs'][int(pk['seq'][i])]['kind']=='native' for i in ids),'multi':sum(D['seqs'][int(pk['seq'][i])]['kind']=='multi' for i in ids),'short_le16':int(((pk['b'][ids]-pk['a'][ids])<=16).sum())}
        reg=np.linalg.lstsq(np.stack([rc,np.ones_like(rc)],axis=1),rn,rcond=None)[0]
        result['heldout'][f'f{h}']={'C':{'auc':auc(rc[p],rc[n]),'neg':stats(rc[n]),'pos':stats(rc[p])},'N':{'auc':auc(rn[p],rn[n]),'neg':stats(rn[n]),'pos':stats(rn[p])},'score_pearson':float(np.corrcoef(rc,rn)[0,1]),'affine_N_from_C':reg.tolist(),'C_positive_N_nonpositive':int(((rc>0)&(rn<=0)).sum()),'N_only_harms':group(extra&n),'N_only_fixes':group(extra&p),'eligible_other_true_classes':int((~(p|n)).sum())}
    result.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),a44_weights_manifest_sha256=hashlib.sha256((root/'weights_s0.json').read_bytes()).hexdigest(),a44_private_readout_cache_sha256=hashlib.sha256((root/'readout_cache_s0.npz').read_bytes()).hexdigest(),cpu_seconds=time.time()-start,finished=t.now())
    out=ROOT/'docs/results/2026-10-28_post_a44_risk_diagnosis';out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(result,indent=2,default=float)+'\n')
    ledger=ROOT/'runs/video_a4c_support_v1/ledger.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    t.ledger_append(ledger,'post-A44 cached risk diagnosis','cpu',started,time.time()-start,True,'No new model or pixel; existing A44 logs/cache only')
    print(json.dumps({'fit':{k:{j:v[j] for j in ['full_fit_H','fit_false_promotions','last_seen_safe_le_minus1_then_final_positive','last_seen_nonpositive_then_final_positive','last_encounter_to_end_updates_final_promotions']} for k,v in result['fit'].items()},'heldout':{k:{j:v[j] for j in ['C_positive_N_nonpositive','N_only_harms','N_only_fixes','score_pearson','affine_N_from_C']} for k,v in result['heldout'].items()},'cpu_seconds':result['cpu_seconds']},default=float))

if __name__=='__main__':main()
