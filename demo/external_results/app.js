(() => {
  const d=window.EXTERNAL_RESULTS;
  if(!d)throw new Error('Verified aggregate data are missing.');
  const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const score=v=>v==null?'—':v.toFixed(3);
  const tr=(cells,ours=false,issue=false)=>'<tr'+(issue?' class="issue"':ours?' class="ours"':'')+'>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
  const fill=(id,rows)=>document.getElementById(id).innerHTML=rows.join('');
  const rev=d.benchmark_review,c=rev.audit.counts;
  const rule={native:'Original complete recording; one propagated clip label',pairs:'Two 1–4 s crops; 60 streams per same/different class × original/gamma 0.9 cell',multi:'4–8 crops of 1–4 s; classes sampled uniformly; same-next probability 0.35; gamma 0.9 probability 0.5'};
  fill('construction-body',['native','pairs','multi'].map(k=>{const r=c[k];return tr([k==='native'?'Native clips':k==='pairs'?'Two-part streams':'Multi-part streams',esc(rule[k]),r.items,r.unique_recordings+' / '+r.sessions,r.true_segments,r.semantic_changes,r.short_true_segments_le16]);}));
  fill('all-models-body',rev.all_models.map(r=>tr([esc(r.name),r.source,pct(r.native),pct(r.pairs),pct(r.multi),score(r.cuts)],r.ours)));
  fill('coverage-body',rev.class_coverage.map(r=>tr([esc(r.name),r.native,r.pair_fragments,r.multi_fragments,r.multi_recordings],false,r.pair_fragments<5)));
  const pairDirections=c.pairs.semantic_transition_counts.flat().filter(v=>v>0).length;
  fill('review-body',[
    tr(['Pair-stream coverage','SC4C: 0 fragments; A5C: 4 fragments; only '+pairDirections+'/20 directed class transitions','Incomplete five-class test; no A4C ↔ A5C transition is tested'],false,true),
    tr(['Multi-stream class mix','A5C: '+c.multi.per_class_fragments.A5C+'/'+c.multi.fragments+' fragments versus 40/800 native clips','Minority classes are deliberately upweighted; pair/multi scores test different distributions']),
    tr(['Artificial joins',c.multi.cross_session_joins+'/'+c.multi.physical_joins+' multi joins cross recording sessions','Controlled splice stress test; not natural same-patient probe transitions']),
    tr(['Short temporal support',c.multi.short_true_segments_le16+'/'+c.multi.true_segments+' true multi segments are ≤1.6 s','Short segments and mixed-view windows can expose boundary/label instability']),
    tr(['Repeated source recordings',c.multi.fragments+' fragments from '+c.multi.unique_recordings+' recordings / '+c.multi.sessions+' sessions','Streams share sources; stream-bootstrap intervals do not establish patient-independent uncertainty']),
    tr(['Ground truth','Clip labels propagated to frames; adjacent same-class fragments merged','No expert frame-level anatomy or natural-transition adjudication']),
    tr(['Test history','Official EV9V test reused in historical rounds; sessions are not verified patients','Exploratory comparison; not an untouched clinical holdout']),
    tr(['Whole-pipeline fairness','Same source frames and labels; different backbones, training and decoder choices','Valid comparison of these frozen implementations; not every original paper recipe or an overall SOTA ranking']),
    tr(['Assessment','Useful for controlled temporal robustness; insufficient as the sole generalization benchmark','Next independent benchmark needs class/transition coverage and expert-labeled real continuous scans'])
  ]);
  fill('recall-body',rev.class_recalls.map(r=>tr([esc(r.name),pct(r.accuracy),...['PLAX','PSAX','A4C','A5C','SC4C'].map(k=>pct(r[k])),r.fp,r.fn])));
  fill('implementation-body',[
    tr(['A08 video Transformer + B3','EchoPrime-initialized video encoder; EV9V task training; frozen s0/s2','16-sample windows every five 10-Hz samples; target marker; frozen B3; no reset at joins']),
    tr(['A14 local router','A08 + trained switch head/router; frozen s0/s2','Routes predicted candidate regions; s2 selected no-op']),
    tr(['EchoViewCLIP stage 1','Stage-1 recipe retrained on EV9V train; CLIP ViT-B/16 initialization','Windowed video recognition; raw nine codes mapped to five families']),
    tr(['STFM','Project numeric-frame-order s100 EV9V checkpoint','A26 sliding-window application; A50 separately uses native whole-file inference']),
    tr(['ASFormer / MS-TCN','Official architectures retrained on EV9V training streams; B0 features','Full-stream temporal labels; not original action-dataset checkpoints']),
    tr(['EchoPrime view classifier','Released ConvNeXt-Base 11-coarse-view weights; no EV9V fine-tuning','Per-frame labels, HMM or validation-selected B3; different from the paper’s 58-view endpoint']),
    tr(['B0 / smoothing / HMM / B3','Legacy frozen EV9V frame encoder and preserved decoder settings','Same corrected numeric-order frames; whole-pipeline controls'])
  ]);
  fill('original-body',[d.original[0],d.original[4]].map(r=>tr(['A50 · original-native rule',esc(r.name),esc(r.seed),r.n,pct(r.accuracy),pct(r.macro_f1),pct(r.a4c),pct(r.a5c)],r.ours)));
  fill('paper-body',d.paper_reported.results.map(r=>tr([esc(r.dataset)+' · '+r.classes+' classes',r.n,esc(r.model),pct(r.accuracy),score(r.f1),score(r.ovr_auc),esc(r.A08_same_protocol),'<a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">'+esc(r.evidence)+' ↗</a>'])));
  const status={executable_shared_labels:'Completed on shared five classes',original_pixels_and_truth_unavailable:'Original test not obtained',institutional_DUA_and_task_mismatch:'Institutional test not obtained',available_but_incompatible:'Available; action task incompatible',access_form_and_static_input:'Access request; static-image input',original_view_test_unavailable:'Original view test not obtained',public_data_different_target:'Public targets differ from view labels',original_test_unavailable_and_target_mismatch:'Test not obtained; box target differs',incompatible_or_exact_test_unverified:'Independent cine truth not verified'};
  fill('access-body',d.availability.models.map(r=>tr([esc(r.model),esc(r.original_test),esc(status[r.status]||r.status),'<a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">Source ↗</a>'])));
  fill('protocol-body',[
    tr(['Official source / labels','888 test recordings; exclude 88 PMPALA; four PSAX subtypes merged → PLAX, PSAX, A4C, A5C, SC4C']),
    tr(['Frame/time axis','Corrected numeric acquisition order; 10 Hz; model-specific original preprocessing; no join identities as input']),
    tr(['Construction seed','Recipe seed 1234; deterministic test offsets +1 (pairs), +1001 (multi); fixed banks retained']),
    tr(['Pair source selection','Disjoint shuffled pools for four nuisance cells; same-label pairing or greedy largest-class pairing; no recording reuse']),
    tr(['Multi source selection','Reuse allowed across streams, never within a stream; only clips with ≥10 sampled frames eligible']),
    tr(['Example true segments','A4C 2s + PLAX 2s + PLAX 3s + A5C 1s → three true segments; adjacent PLAX fragments merge']),
    tr(['Classification macro-F1','One decoded majority label per native clip, then equal-weight mean of five class F1 values']),
    tr(['Segment F1@50','Same-class one-to-one temporal segments with IoU ≥0.5; pooled counts; not classification macro-F1']),
    tr(['A26 versus A50','A26 is windowed stream inference; A50 uses STFM’s native 10-key whole-file rule and retains unsupported predictions as errors']),
    tr(['Training overlap','Available EV9V training manifests show zero test clip/session intersections; external pretraining and patient overlap remain unknown']),
    tr(['Audit','Private manifest and all nine A26 native label sets independently recounted; five H09/A26 shared rows match exactly; no model forwards']),
    tr(['Original-paper rows','Published values; original pixels/complete labels not obtained; A08 has five outputs, not 38/58']),
    tr(['Downloads','<a href="results.json" download>JSON ↗</a> &nbsp; <a href="results.csv" download>CSV ↗</a>'])
  ]);
})();
