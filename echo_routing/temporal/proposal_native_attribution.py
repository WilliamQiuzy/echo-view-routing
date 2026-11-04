"""Cached native frame attribution; propagated references, no clinical adjudication."""
import numpy as np
from echo_routing.temporal.anchor_edits import aligned,counts,predicted_count
from echo_routing.temporal.proposal_features import short_fp
from echo_routing.temporal.segments import labels_to_segments


def joint_counts(truth,base,pred):
    truth,base=aligned(truth,base);truth,pred=aligned(truth,pred);bc=base==truth;pc=pred==truth
    cube=np.bincount(25*truth+5*base+pred,minlength=125).reshape(5,5,5)
    return {'true_Base_prediction_joint5cube':cube.tolist(),'positions':len(truth),'Base_correct_to_wrong':int(np.count_nonzero(bc & ~pc)),'Base_wrong_to_correct':int(np.count_nonzero(~bc & pc)),'both_correct':int(np.count_nonzero(bc & pc)),'both_wrong_unchanged_class':int(np.count_nonzero(~bc & ~pc & (base==pred))),'both_wrong_changed_class':int(np.count_nonzero(~bc & ~pc & (base!=pred))),'changed_positions':int(np.count_nonzero(base!=pred)),'Base_frame_errors':int(np.count_nonzero(~bc)),'new_frame_errors':int(np.count_nonzero(~pc)),'frame_error_delta':int(np.count_nonzero(~pc)-np.count_nonzero(~bc))}


def effects(base,pred,truth,fit_f1):
    truth,base=aligned(truth,base);truth,pred=aligned(truth,pred)
    if not np.all(truth==truth[0]):raise ValueError('native constant propagated reference required')
    before=counts(base,truth);after=counts(pred,truth);joint=joint_counts(truth,base,pred)
    c=int(truth[0]);old=labels_to_segments(base);new=labels_to_segments(pred);extended=created=0;cross=False;long_wrong=False
    for ps in new:
        if ps.label==c:continue
        added=int(np.count_nonzero((base[ps.start:ps.end]==truth[ps.start:ps.end]) & (pred[ps.start:ps.end]!=truth[ps.start:ps.end])))
        if not added:continue
        seeds=[bs for bs in old if bs.label==ps.label and min(bs.end,ps.end)>max(bs.start,ps.start)]
        if seeds:extended+=added
        else:created+=added
        if ps.length>16:
            long_wrong=True;cross|=any(bs.length<=16 for bs in seeds)
    dm,dp=after[0]-before[0],after[1]-before[1]
    return {'delta_M':dm,'delta_P':dp,'u':2*dm-fit_f1*dp,'dE':joint['frame_error_delta'],'dFP16':short_fp(pred,truth)-short_fp(base,truth),'added_cuts':predicted_count(pred)-predicted_count(base),'newly_wrong_positions_extending_old_wrong_run':extended,'newly_wrong_positions_new_wrong_segment':created,'newly_wrong_positions_in_long_wrong_segments':sum(int(np.count_nonzero((base[s.start:s.end]==truth[s.start:s.end]) & (pred[s.start:s.end]!=truth[s.start:s.end]))) for s in new if s.label!=c and s.length>16),'wrong_run_crossed_short16_to_long':bool(cross),'new_wrong_positions_touch_long_wrong_segment':bool(long_wrong)}
