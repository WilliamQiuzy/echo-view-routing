"""Fixed same-score MAP readout with checksum-verified integer label prefixes."""
import torch
from echo_routing.temporal.structured_predictions import LabelReadout
from echo_routing.temporal.dense_readout import sha

@torch.no_grad()
def decode_bank(model,bank,folder,identity,on_progress=None):
    cache=LabelReadout(folder,identity,[len(q['labels']) for q in bank.sequences]);initial=cache.cursor
    for i in range(initial,len(bank.sequences)):
        a,b,_=bank.values[i];cache.run_one(i,lambda a=a,b=b:model.decode(a,b).cpu().numpy())
        if on_progress is not None and (i+1)%100==0:on_progress(i+1,len(bank.sequences))
    predictions={q['id']:cache.load(i) for i,q in enumerate(bank.sequences)}
    return predictions,{'identity':identity,'sequences':cache.cursor,'new_decodes':cache.cursor-initial,'journal_sha256':sha(cache.journal),'original_positions':sum(len(q['labels']) for q in bank.sequences),'truth_passed_to_model':False}
