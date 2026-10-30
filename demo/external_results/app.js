(() => {
  const d=window.EXTERNAL_RESULTS;
  if(!d)throw new Error('Verified aggregate data are missing.');
  const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const score=v=>v==null?'—':v.toFixed(3);
  const tr=(cells,ours=false)=>'<tr'+(ours?' class="ours"':'')+'>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
  const fill=(id,rows)=>document.getElementById(id).innerHTML=rows.join('');
  fill('original-body',[d.original[0],d.original[4]].map(r=>tr(['EV9V · five classes',esc(r.name),esc(r.seed),r.n,pct(r.accuracy),pct(r.macro_f1),pct(r.a4c),pct(r.a5c)],r.ours)));
  fill('class-body',d.class_breakdown.map(r=>tr([esc(r.name),r.n,pct(r.a08_f1),pct(r.stfm_f1)])));
  const pp=v=>(v>=0?'+':'')+(v*100).toFixed(2);
  const pair=d.paired_original[0];
  fill('paired-body',[
    tr(['Accuracy',pp(pair.accuracy_difference),pair.accuracy_difference_ci95.map(pp).join(' to ')]),
    tr(['Macro-F1',pp(pair.macro_f1_difference),pair.macro_f1_difference_ci95.map(pp).join(' to ')])
  ]);
  fill('history-body',d.history.map(r=>tr(['A26 · constructed streams',esc(r.name),pct(r.native),pct(r.pairs),pct(r.multi)],r.ours)));
  fill('paper-body',d.paper_reported.results.map(r=>tr([esc(r.dataset)+' · '+r.classes+' classes',r.n,esc(r.model),pct(r.accuracy),score(r.f1),score(r.ovr_auc),esc(r.A08_same_protocol),'<a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">'+esc(r.evidence)+' ↗</a>'])));
  const status={executable_shared_labels:'Completed on shared five classes',original_pixels_and_truth_unavailable:'Original test not obtained',institutional_DUA_and_task_mismatch:'Institutional test not obtained',available_but_incompatible:'Available; action task incompatible',access_form_and_static_input:'Access request; static-image input',original_view_test_unavailable:'Original view test not obtained',public_data_different_target:'Public targets differ from view labels',original_test_unavailable_and_target_mismatch:'Test not obtained; box target differs',incompatible_or_exact_test_unverified:'Independent cine truth not verified'};
  fill('access-body',d.availability.models.map(r=>tr([esc(r.model),esc(r.original_test),esc(status[r.status]||r.status),'<a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">Source ↗</a>'])));
  fill('protocol-body',[
    tr(['Matched native comparison','A50 · 800 recordings / 125 sessions · shared five classes']),
    tr(['A08 score aggregation','Mean of three frozen-model scores; one majority-vote label per recording']),
    tr(['STFM version','Project reproduction · legacy frozen s100 weights · native 10-key-frame inference']),
    tr(['Paired uncertainty','2,000 bootstrap resamples of recording sessions']),
    tr(['Historical A26 seeds','A08 s0/s2 mean; one frozen version per external model']),
    tr(['Independent external five-class confirmation','Not established; EV9V was already used in this project']),
    tr(['Original-paper rows','Author-reported values; A08 has not been evaluated on those test protocols']),
    tr(['Downloads','<a href="results.json" download>JSON ↗</a> &nbsp; <a href="results.csv" download>CSV ↗</a>'])
  ]);
})();
