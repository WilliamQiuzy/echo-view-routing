"""Historical full-task metrics and explicitly paired duration-shift replacement."""
from types import SimpleNamespace
import numpy as np
from echo_routing.evaluate.dev_metrics import evaluate_predictions
from echo_routing.temporal import dense_runtime as data
from echo_routing.temporal.dense_analysis import short_metrics
from echo_routing.evaluate.failure_units import session_of

def stress_context(main,stress,expected_streams=120):
    if len(stress)!=expected_streams or any(q['kind']!='multi' for q in stress) or {q['id'] for q in stress}&{q['id'] for q in main['sequences']}:raise ValueError('fixed new multi-only stress bank required')
    return {'sequences':[q for q in main['sequences'] if q['kind']!='multi']+stress,'eval_set':SimpleNamespace(natives=main['eval_set'].natives,pairs=main['eval_set'].pairs,multi=[data.metric_stream(q) for q in stress])}

def compose_stress_predictions(main_predictions,stress_predictions,main_sequences,stress_sequences):
    if set(main_predictions)!={q['id'] for q in main_sequences} or set(stress_predictions)!={q['id'] for q in stress_sequences}:raise ValueError('complete original and stress identities required')
    return {**{q['id']:main_predictions[q['id']] for q in main_sequences if q['kind']!='multi'},**stress_predictions}

def evaluate(predictions,context):
    qs=context['sequences']
    if set(predictions)!={q['id'] for q in qs}:raise ValueError('complete fixed task prediction coverage required')
    for q in qs:
        value=np.asarray(predictions[q['id']])
        if value.shape!=q['labels'].shape or value.dtype.kind not in 'iu' or (value<0).any() or (value>=5).any():raise ValueError('complete original-axis integer five-class predictions required')
    result,tables=evaluate_predictions(predictions,context['eval_set'],10,.5);subset=[q for q in qs if q['kind'] in ['native','multi']];extra,_=data.metrics(subset,{q['id']:predictions[q['id']] for q in subset});result.update({k:v for k,v in extra.items() if k.startswith('short_') or k.startswith('native_A4C') or k.startswith('native_A5C')});result.update(short_metrics(qs,predictions,16));conf=np.asarray(result['native_confusion']);den=conf.sum(0)+conf.sum(1);result['native_per_class_f1']=np.divide(2*np.diag(conf),den,out=np.zeros(5,float),where=den>0).tolist();position={name:np.zeros((5,5),np.int64) for name in ['native','multi','pair']}
    for q in qs:np.add.at(position['native' if q['kind']=='native' else 'multi' if q['kind']=='multi' else 'pair'],(q['labels'],predictions[q['id']]),1)
    result['decoded_position_confusion']={k:v.tolist() for k,v in position.items()};result['counts']['native_sessions']=len({session_of(q['id']) for q in qs if q['kind']=='native'})
    for kind,name in [('multi','multi_seg'),('pair','pairs_seg')]:
        s=tables[name].sum(numeric_only=True);result['counts'].update({f'{kind}_{suffix}':int(s[column]) for suffix,column in [('predicted_segments','seg_pred'),('true_segments','seg_true'),('matched_segments','seg_matched'),('positions','n')]})
    result['counts']['native_positions']=int(tables['native'].n.sum());result['short_segment_count_scope']='multi_only;IoU50classawareone_to_one';return result,tables

def verify_reference(result,expected):
    def visit(actual,want):
        if isinstance(want,dict):
            for k,v in want.items():visit(actual[k],v)
        elif isinstance(want,float):
            if not np.isclose(actual,want,rtol=0,atol=1e-12):raise ValueError('fixed historical reference metric did not reproduce')
        elif actual!=want:raise ValueError('fixed historical reference metric did not reproduce')
    visit(result,expected)
