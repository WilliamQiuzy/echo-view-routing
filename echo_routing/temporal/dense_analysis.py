"""Cache-only diagnostics using the unchanged class-aware IoU50 matching."""
from echo_routing.temporal.segments import labels_to_segments
from echo_routing.evaluate.failure_units import match_segments

def short_metrics(sequences,predictions,maximum):
 out={f'short_true_segments_{maximum}samples':0,f'short_true_unmatched_{maximum}samples':0,f'short_predicted_segments_{maximum}samples':0,f'short_spurious_segments_{maximum}samples':0}
 for q in sequences:
  if q['kind'] not in ['multi','mixed']:continue
  true=labels_to_segments(q['labels']);pred=labels_to_segments(predictions[q['id']]);matched=match_segments(pred,true);used=set(matched.values())
  out[f'short_true_segments_{maximum}samples']+=sum(t.end-t.start<=maximum for t in true);out[f'short_true_unmatched_{maximum}samples']+=sum(t.end-t.start<=maximum and i not in matched for i,t in enumerate(true));out[f'short_predicted_segments_{maximum}samples']+=sum(t.end-t.start<=maximum for t in pred);out[f'short_spurious_segments_{maximum}samples']+=sum(t.end-t.start<=maximum and i not in used for i,t in enumerate(pred))
 return out


def validate_inner_cache(probabilities,sequences,expected_input_checksum,checkpoint_record):
 import numpy as np
 from echo_routing.temporal.dense_runtime import metrics
 if checkpoint_record.get('input_sequence_sha256')!=expected_input_checksum or set(probabilities)!={q['id'] for q in sequences}:raise ValueError('inner cache input checksum/identity mismatch')
 for q in sequences:
  p=np.asarray(probabilities[q['id']])
  if p.shape!=(len(q['labels']),5) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('inner cache shape/finite/normalization mismatch')
 result,_=metrics(sequences,{k:np.asarray(p).argmax(1) for k,p in probabilities.items()})
 if any(checkpoint_record.get(k)!=v for k,v in result.items()):raise ValueError('cached raw metrics differ from checkpoint inner record')
 return result
