(() => {
  const d=window.EXTERNAL_RESULTS;
  if(!d){document.getElementById('summary').textContent='已核验的汇总文件尚未生成。';return;}
  const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const dec=v=>v==null?'—':Number(v).toFixed(3);
  const ci=a=>a?'['+a.map(pct).join(', ')+']':'—';
  const tr=(cells,ours=false)=>'<tr'+(ours?' class="ours"':'')+'>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
  const fill=(id,rows)=>document.getElementById(id).innerHTML=rows.join('');
  document.getElementById('summary').textContent=d.summary;
  fill('external-body',d.external.map(r=>tr([esc(r.name)+'<small>'+esc(r.note)+'</small>',esc(r.label)+' / '+r.n,pct(r.mean_recall)+'<small>'+ci(r.ci95)+'</small>',r.correct.join(' / ')+'<small>每个种子分母均为 '+r.n+'</small>',dec(r.cuts_per_clip)],true)));
  fill('seed-body',d.seed_rows.map(r=>tr([esc(r.dataset),esc(r.decoder),r.seed,pct(r.recall),dec(r.cuts_per_clip)])));
  fill('original-body',d.original.map(r=>tr([esc(r.name),esc(r.seed),r.n,pct(r.accuracy),pct(r.macro_f1)+'<small>'+ci(r.macro_ci95)+'</small>',pct(r.a4c)+' / '+pct(r.a5c)],r.ours)));
  fill('original-detail-body',d.original_details.map(r=>tr([esc(r.name),esc(r.population),pct(r.accuracy),pct(r.macro_f1)],r.ours)));
  const pp=v=>(v>=0?'+':'')+(v*100).toFixed(2);
  const delta=(v,c)=>pp(v)+'<small>['+c.map(pp).join(', ')+']</small>';
  fill('paired-body',d.paired_original.map(r=>tr([esc(r.name),delta(r.accuracy_difference,r.accuracy_difference_ci95),delta(r.macro_f1_difference,r.macro_f1_difference_ci95)])));
  fill('history-body',d.history.map(r=>tr([esc(r.name),pct(r.native),pct(r.pairs),pct(r.multi),dec(r.fp)+'<small>'+dec(r.fp_short)+'</small>',dec(r.fn)+'<small>'+dec(r.fn_short)+'</small>'],r.ours)));
  const status={executable_shared_labels:'已运行 · 共同标签',original_pixels_and_truth_unavailable:'原像素 / 真值未取得',institutional_DUA_and_task_mismatch:'机构数据 / 任务不同',available_but_incompatible:'可获取 · 任务不兼容',access_form_and_static_input:'申请入口 / 静态输入',original_view_test_unavailable:'原切面test未取得',public_data_different_target:'公开数据 · 原目标不同',original_test_unavailable_and_target_mismatch:'未取得 / 目标不同',incompatible_or_exact_test_unverified:'目标不同 / 未核实'};
  fill('access-body',d.availability.models.map(r=>tr([esc(r.model),esc(r.original_test),'<span class="status '+(r.status==='executable_shared_labels'?'done':'')+'">'+esc(status[r.status]||r.status)+'</span>',esc(r.limit)+' <a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">来源 ↗</a>'])));
  const p=document.getElementById('resource-notes');
  p.innerHTML=d.availability.additional_resources.map(r=>'<p><a href="'+esc(r.source)+'" target="_blank" rel="noopener noreferrer">'+esc(r.name)+'</a>：'+esc(r.disposition)+'</p>').join('')+'<p>模型：'+esc(d.model_identity)+'</p>';
  document.getElementById('verification-note').textContent=d.verification_note;
  const e=d.resources;
  fill('execution-body',[
    tr(['A08固定三个种子',e.source_sequences.toLocaleString()+' 条 × 3','无重新训练；全部五类输出保留']),
    tr(['STFM原生整段推理',e.baseline_sequences+' 条','九码888 / 共同五类800；两种汇总规则']),
    tr(['L40S作业 / 实测前向',(e.gpu_job_elapsed_seconds/60).toFixed(2)+' / '+(e.gpu_forward_seconds/60).toFixed(2)+' 分钟','包含LVH前向前退出的首次作业']),
    tr(['独立CPU重算','2,470条缓存 · 2,000次重采样','完整标签/计数/指标核对通过；未重新加载原始像素']),
    tr(['新增服务器运行文件',(e.run_storage_bytes/1e6).toFixed(2)+' MB','不含已有原始数据/模型与网页副本'])
  ]);
})();
