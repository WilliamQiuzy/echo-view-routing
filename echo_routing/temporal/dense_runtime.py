"""A57 raw sequence IO and adapter to the existing segmentation metrics."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
import torch
from echo_routing.temporal.dense_sequence import chunk_positions,chunk_starts,stitch_probabilities
from echo_routing.config import load_paths
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.temporal.joint_visual import load_frame_u8
from echo_routing.evaluate.dev_metrics import native_table,_clip_summary,stream_tables
from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments


def verify_partition(sequences,allow):
    seen=set()
    for q in sequences:
        if q['id'] in seen or len(q['labels'])!=len(q['refs']) or not len(q['labels']):raise ValueError('duplicate identity or label/reference length mismatch')
        seen.add(q['id'])
        if not {r[0] for r in q['refs']}<=set(allow):raise ValueError('full sequence footprint leaves allowed partition')
        if np.any((q['labels']<0)|(q['labels']>4)):raise ValueError('five-class labels required')


def chunk_arrays(sequence,start):return chunk_positions(len(sequence['labels']),start)


class Pixels:
    def __init__(self,sequences):
        needed={r[0] for q in sequences for r in q['refs']}
        with load_paths().manifest_path.open() as f:self.dirs={r['video_id']:r['frames_dir'] for r in csv.DictReader(f) if r['video_id'] in needed}
        if set(self.dirs)!=needed:raise ValueError('missing frame directory for allowlisted source')
        self.paths={}
    @lru_cache(maxsize=512)
    def frame(self,video,index,gamma):
        if video not in self.paths:self.paths[video]=list_frame_paths(Path(self.dirs[video]))
        paths=self.paths[video];return load_frame_u8(paths[min(int(index),len(paths)-1)],round(float(gamma),6))
    def chunk(self,q,start):
        indices,mask=chunk_arrays(q,start)
        frames=np.stack([self.frame(*q['refs'][int(i)]) for i in indices])
        return frames,np.asarray(q['labels'][indices],np.int64),mask
    def batch(self,sequences,starts,device):
        chunks=[self.chunk(q,int(start)) for q,start in zip(sequences,starts)]
        return tuple(torch.from_numpy(np.stack([row[i] for row in chunks])).to(device) for i in range(3))


@torch.no_grad()
def predict_sequence(model,q,pixels,device='cuda'):
    starts=chunk_starts(len(q['labels']));chunks=[]
    for offset in range(0,len(starts),2):
        ss=starts[offset:offset+2];b=pixels.batch([q]*len(ss),ss,device);chunks.extend(model.probabilities(b[0],b[2]).cpu().numpy())
    return stitch_probabilities(chunks,len(q['labels'])).astype(np.float32)


def metric_stream(q):
    edges=[0,*map(int,q['joins']),len(q['labels'])];fragments=[]
    for start,end in zip(edges[:-1],edges[1:]):
        fragments.append(SimpleNamespace(video_id=q['refs'][start][0],label=int(q['labels'][start]),variant='orig' if q['refs'][start][2]==1 else 'gamma',start=0,end=end-start))
    joins=np.asarray(q['joins'],int);sem=np.array([j for j in joins if q['labels'][j]!=q['labels'][j-1]],int)
    return SimpleNamespace(recipe=SimpleNamespace(recipe_id=q['id'],cell='A57_inner',fragments=fragments),labels=q['labels'],joins=joins,semantic_boundaries=sem)


def metrics(sequences,predictions):
    if set(predictions)!={q['id'] for q in sequences}:raise ValueError('prediction identity coverage mismatch')
    if any(np.asarray(predictions[q['id']]).shape!=q['labels'].shape or np.any((np.asarray(predictions[q['id']])<0)|(np.asarray(predictions[q['id']])>4)) for q in sequences):raise ValueError('complete original timeline five-class predictions required')
    native=[SimpleNamespace(seq_id=q['id'],labels=q['labels']) for q in sequences if q['kind']=='native'];mixed=[metric_stream(q) for q in sequences if q['kind']!='native']
    nat=native_table(predictions,native);seg,frag=stream_tables(predictions,mixed,tol=5);s=seg.sum(numeric_only=True);f=frag.sum(numeric_only=True)
    ratio=lambda a,b:float(a/b) if b else None
    cm=_clip_summary(nat,5);conf=np.asarray(cm['native_confusion']);extra={'short_true_segments_10samples':0,'short_true_unmatched_10samples':0,'short_predicted_segments_10samples':0,'short_spurious_segments_10samples':0}
    for q in sequences:
        if q['kind']=='native':continue
        ps,ts=labels_to_segments(predictions[q['id']]),labels_to_segments(q['labels']);matched=match_segments(ps,ts);used=set(matched.values())
        extra['short_true_segments_10samples']+=sum(t.end-t.start<=10 for t in ts);extra['short_true_unmatched_10samples']+=sum(t.end-t.start<=10 and i not in matched for i,t in enumerate(ts));extra['short_predicted_segments_10samples']+=sum(p.end-p.start<=10 for p in ps);extra['short_spurious_segments_10samples']+=sum(p.end-p.start<=10 and i not in used for i,p in enumerate(ps))
    result={**cm,'native_frame_accuracy':ratio(nat.frame_correct.sum(),nat.n.sum()),'native_internal_cuts_per_clip':float(nat.cuts.mean()),'native_internal_cuts_per_min':ratio(nat.cuts.sum(),nat.n.sum()/600),'native_A4C_to_A5C':int(conf[2,3]),'native_A5C_to_A4C':int(conf[3,2]),'native_A4C_A5C_errors':int(conf[2,3]+conf[3,2]),'segment_f1_50_multi':ratio(2*s.seg_matched,s.seg_pred+s.seg_true),'boundary_recall_multi':ratio(f.t_matched,f.t_true),'boundary_precision_multi':ratio(f.t_matched,f.p_pred),'boundary_f1_multi':ratio(2*f.t_matched,f.t_true+f.p_pred),'same_class_cut_rate_multi':ratio(f.false_split,f.same_join),'multi_frame_accuracy':ratio(s.correct,s.n),'counts':{'native_clips':len(nat),'multi_streams':len(seg),'native_positions':int(nat.n.sum()),'multi_positions':int(s.n),'multi_true_changes':int(f.t_true),'multi_same_joins':int(f.same_join),'multi_predicted_segments':int(s.seg_pred),'multi_true_segments':int(s.seg_true),'multi_matched_segments':int(s.seg_matched)},**extra}
    return result,{'native':nat,'multi_seg':seg,'multi_frag':frag}


def choose_inner(records):
    return max(records,key=lambda r:(r['segment_f1_50_multi'],r['native_macro_f1'],-r['native_A4C_A5C_errors'],-r.get('backbone_lr',0),-r.get('step',0)))
