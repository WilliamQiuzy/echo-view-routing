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
