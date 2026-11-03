"""Paired complete-session/stream/pair bootstrap, conditional on the fixed banks."""
import numpy as np
from echo_routing.evaluate.failure_units import session_of,match_segments
from echo_routing.temporal.segments import labels_to_segments

def prepare_system(predictions,sequences,tables):
    sessions={};native={};streams={'multi':[],'pair':[]};matched={'multi':{},'pair':{}};truths={}
    for q in sequences:
        y=np.asarray(q['labels']);p=np.asarray(predictions[q['id']])
        if q['kind']=='native':
            lab=int(y[0]);majority=int(np.bincount(p,minlength=5).argmax());native[q['id']]=(lab,majority);session=session_of(q['id']);c,e=sessions.setdefault(session,(np.zeros((5,5),np.int64),np.zeros(4,np.int64)));c[lab,majority]+=1;e+=np.array([np.count_nonzero(p[1:]!=p[:-1]),len(p),np.count_nonzero(p==y),1]);continue
        kind='multi' if q['kind']=='multi' else 'pair';true=labels_to_segments(y);pred=labels_to_segments(p);ma=match_segments(pred,true);used=set(ma.values());matched[kind][q['id']]=set(ma);truths[q['id']]=[t.end-t.start for t in true];fr=tables['multi_frag' if kind=='multi' else 'pairs_frag'];f=fr[fr.stream==q['id']].sum(numeric_only=True);stats=[len(ma),len(pred),len(true),len(y),np.count_nonzero(p==y),f.t_matched,f.t_true,f.p_pred,f.false_split,f.same_join]
        for maximum in [10,16]:stats.extend([sum(n<=maximum for n in truths[q['id']]),sum(n<=maximum and i not in ma for i,n in enumerate(truths[q['id']])),sum(s.end-s.start<=maximum for s in pred),sum(s.end-s.start<=maximum and i not in used for i,s in enumerate(pred))])
        streams[kind].append((q['id'],np.asarray(stats,np.int64)))
    return {'session_order':list(sessions),'session_confusion':np.asarray([v[0] for v in sessions.values()]),'session_extra':np.asarray([v[1] for v in sessions.values()]),'native':native,'streams':streams,'matched':matched,'true_lengths':truths}

def _weights(rng,n,n_boot):
    if not n:raise ValueError('positive declared unit count required')
    indices=rng.integers(0,n,(n_boot,n));return np.bincount((np.arange(n_boot)[:,None]*n+indices).ravel(),minlength=n_boot*n).reshape(n_boot,n)

def _ratio(a,b,fill=np.nan):return np.divide(a,b,out=np.full_like(a,fill,dtype=float),where=b!=0)
def _macro(c):return _ratio(2*np.diagonal(c,axis1=-2,axis2=-1),c.sum(-1)+c.sum(-2),fill=0).mean(-1)

def paired_bootstrap(left,right,n_boot=5000,seed=2026110358):
    if left['session_order']!=right['session_order'] or set(left['native'])!=set(right['native']) or left['true_lengths']!=right['true_lengths']:raise ValueError('paired units/truth identity differs')
    rng=np.random.default_rng(seed);delta={};w=_weights(rng,len(left['session_order']),n_boot);a=(w@left['session_confusion'].reshape(len(w[0]),25)).reshape(n_boot,5,5);b=(w@right['session_confusion'].reshape(len(w[0]),25)).reshape(n_boot,5,5);delta['native_macro_f1']=_macro(a)-_macro(b);ae=w@left['session_extra'];be=w@right['session_extra'];delta['native_internal_cuts_per_clip']=_ratio(ae[:,0],ae[:,3])-_ratio(be[:,0],be[:,3]);delta['native_frame_accuracy']=_ratio(ae[:,2],ae[:,1])-_ratio(be[:,2],be[:,1]);delta['native_accuracy']=_ratio(np.trace(a,axis1=1,axis2=2),a.sum((1,2)))-_ratio(np.trace(b,axis1=1,axis2=2),b.sum((1,2)))
    for kind in ['multi','pair']:
        la,lb=left['streams'][kind],right['streams'][kind]
        if [r[0] for r in la]!=[r[0] for r in lb]:raise ValueError('whole-stream/pair order differs')
        w=_weights(rng,len(la),n_boot);a=w@np.stack([r[1] for r in la]);b=w@np.stack([r[1] for r in lb]);suffix='multi' if kind=='multi' else 'pairs';delta['segment_f1_50_'+suffix]=_ratio(2*a[:,0],a[:,1]+a[:,2])-_ratio(2*b[:,0],b[:,1]+b[:,2]);delta['boundary_recall_'+suffix]=_ratio(a[:,5],a[:,6])-_ratio(b[:,5],b[:,6]);delta['boundary_precision_'+suffix]=_ratio(a[:,5],a[:,7])-_ratio(b[:,5],b[:,7])
        if kind=='multi':
            delta['same_class_cut_rate_multi']=_ratio(a[:,8],a[:,9])-_ratio(b[:,8],b[:,9]);delta['multi_frame_accuracy']=_ratio(a[:,4],a[:,3])-_ratio(b[:,4],b[:,3])
            for maximum,base in [(10,10),(16,14)]:delta[f'short_true_unmatched_{maximum}samples']=a[:,base+1]-b[:,base+1];delta[f'short_spurious_segments_{maximum}samples']=a[:,base+3]-b[:,base+3]
    exact={'native_left_only_correct':0,'native_right_only_correct':0,'native_both_wrong':0,'native_both_correct':0}
    for key,(lab,a) in left['native'].items():
        other,b=right['native'][key]
        if lab!=other:raise ValueError('native paired class differs')
        x,y=a==lab,b==lab;name='native_both_correct' if x and y else 'native_both_wrong' if not x and not y else 'native_left_only_correct' if x else 'native_right_only_correct';exact[name]+=1
    for kind in ['multi','pair']:
        for label in ['recovered','lost']:
            exact[f'{kind}_true_segments_{label}']=0
            for maximum in [10,16]:exact[f'{kind}_short_true_segments_{label}_{maximum}samples']=0
        for key,a in left['matched'][kind].items():
            b=right['matched'][kind][key]
            for label,indices in [('recovered',a-b),('lost',b-a)]:
                exact[f'{kind}_true_segments_{label}']+=len(indices)
                for maximum in [10,16]:exact[f'{kind}_short_true_segments_{label}_{maximum}samples']+=sum(left['true_lengths'][key][i]<=maximum for i in indices)
    finite={k:np.isfinite(v) for k,v in delta.items()};return {'delta95':{k:np.quantile(v[finite[k]],[.025,.975]).tolist() if finite[k].any() else None for k,v in delta.items()},'undefined_bootstrap_replicates':{k:int((~v).sum()) for k,v in finite.items()},'bootstrap':{'replicates':n_boot,'seed':seed,'native_sessions':len(left['session_order']),'multi_streams':len(left['streams']['multi']),'pair_streams':len(left['streams']['pair']),'units':'complete sessions / complete fixed streams / complete fixed pairs','scope':'conditionalconstructedbankswithsharedsources; sessionsnotconfirmedpatients; noindependentclinicalpopulationclaim','undefined_ratios':'reportedasnullifallundefined;otherwiseintervalconditionalonfinitepairedreplicateswithcountsdisclosed','short_count_intervals':'differencesinresampledwhole-banktotals;pairedtruedenominatorsidenticalineachreplicate'},'exact_disagreements':exact}
