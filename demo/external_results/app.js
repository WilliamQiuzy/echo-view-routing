(() => {
  const d=window.EXTERNAL_RESULTS;
  if(!d){document.getElementById('summary').textContent='Verified aggregate data have not been packaged.';return;}
  const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const dec=v=>v==null?'—':Number(v).toFixed(3);
  const ci=a=>a?'['+a.map(pct).join(', ')+']':'—';
  const tr=(cells,ours=false)=>'<tr'+(ours?' class="ours"':'')+'>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
  const fill=(id,rows)=>document.getElementById(id).innerHTML=rows.join('');
  document.getElementById('summary').textContent=d.summary;
  fill('external-body',d.external.map(r=>tr([esc(r.name)+'<small>'+esc(r.note)+'</small>',esc(r.label)+' / '+r.n,pct(r.mean_recall)+'<small>'+ci(r.ci95)+'</small>',r.correct.join(' / ')+'<small>Denominator per seed: '+r.n+'</small>',dec(r.cuts_per_clip)],true)));
  fill('seed-body',d.seed_rows.map(r=>tr([esc(r.dataset),esc(r.decoder),r.seed,pct(r.recall),dec(r.cuts_per_clip)])));
  fill('original-body',d.original.map(r=>tr([esc(r.name),esc(r.seed),r.n,pct(r.accuracy),pct(r.macro_f1)+'<small>'+ci(r.macro_ci95)+'</small>',pct(r.a4c)+' / '+pct(r.a5c)],r.ours)));
  fill('original-detail-body',d.original_details.map(r=>tr([esc(r.name),esc(r.population),pct(r.accuracy),pct(r.macro_f1)],r.ours)));
  fill('class-body',d.class_breakdown.map(r=>tr([esc(r.name),r.n,pct(r.share),pct(r.a08_precision),pct(r.a08_recall),pct(r.a08_f1),pct(r.stfm_f1)])));
  const x=d.metric_example_s0_A5C;
  document.getElementById('metric-example').textContent='Concrete example, A08 seed 0: '+x.n+' true A5C recordings; '+x.predicted+' predicted as A5C; '+x.correct+' correct. A5C precision is '+pct(x.precision)+', recall '+pct(x.recall)+', and F1 '+pct(x.f1)+'. This seed classifies 782/800 recordings correctly (97.75%), but its five-class macro-F1 is 94.85%. False A5C predictions affect precision even when A5C recall is high.';
  const pp=v=>(v>=0?'+':'')+(v*100).toFixed(2);
  const delta=(v,c)=>pp(v)+'<small>['+c.map(pp).join(', ')+']</small>';
  fill('paired-body',d.paired_original.map(r=>tr([esc(r.name),delta(r.accuracy_difference,r.accuracy_difference_ci95),delta(r.macro_f1_difference,r.macro_f1_difference_ci95)])));
  fill('history-body',d.history.map(r=>tr([esc(r.name),pct(r.native),pct(r.pairs),pct(r.multi),dec(r.fp)+'<small>'+dec(r.fp_short)+'</small>',dec(r.fn)+'<small>'+dec(r.fn_short)+'</small>'],r.ours)));
  const status={executable_shared_labels:'Run · shared labels',original_pixels_and_truth_unavailable:'Original pixels / truth not obtained',institutional_DUA_and_task_mismatch:'Institutional data / different task',available_but_incompatible:'Available · incompatible task',access_form_and_static_input:'Access request / static images',original_view_test_unavailable:'Original view test not obtained',public_data_different_target:'Public data · different target',original_test_unavailable_and_target_mismatch:'Not obtained / different target',incompatible_or_exact_test_unverified:'Different target / exact test unverified'};
  fill('access-body',d.availability.models.map(r=>tr([esc(r.model),esc(r.original_test),'<span class="status '+(r.status==='executable_shared_labels'?'done':'')+'">'+esc(status[r.status]||r.status)+'</span>',esc(r.limit)+' <a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">Source ↗</a>'])));
  document.getElementById('resource-notes').innerHTML=d.availability.additional_resources.map(r=>'<p><a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">'+esc(r.name)+'</a>: '+esc(r.disposition)+'</p>').join('')+'<p>'+esc(d.model_identity)+'</p>';
  document.getElementById('verification-note').textContent=d.verification_note;
  const e=d.resources;
  fill('execution-body',[
    tr(['A08, three frozen seeds',e.source_sequences.toLocaleString()+' recordings × 3','No retraining; all five outputs retained']),
    tr(['STFM native whole-video inference',e.baseline_sequences+' recordings','Nine-code 888 / shared-five 800; two fusion rules']),
    tr(['L40S jobs / measured forward',(e.gpu_job_elapsed_seconds/60).toFixed(2)+' / '+(e.gpu_forward_seconds/60).toFixed(2)+' min','Includes the first job stopped before LVH inference']),
    tr(['Independent CPU recount','2,470 caches · 2,000 bootstrap draws','Full labels, counts and metrics matched; no raw-pixel reload']),
    tr(['New server run files',(e.run_storage_bytes/1e6).toFixed(2)+' MB','Excludes existing data/models and page copies'])
  ]);
})();
