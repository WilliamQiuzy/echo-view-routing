"""Complete cached posteriors only: no frame paths, media or outer data access."""
import numpy as np
import torch
from echo_routing.temporal.structured_inputs import check_probabilities

class PosteriorBank:
    def __init__(self,sequences,a08,jepa,device='cpu'):
        self.sequences=sequences;self.device=device;self.values=[];self.source_identities={'a08':a08.identity,'jepa':jepa.identity}
        if a08.cursor!=len(sequences) or jepa.cursor!=len(sequences):raise ValueError('both frozen posterior prefixes must be complete')
        if len({q['id'] for q in sequences})!=len(sequences):raise ValueError('unique whole-sequence identities required')
        for i,q in enumerate(sequences):
            labels=np.asarray(q['labels']);n=len(q['refs'])
            if labels.shape!=(n,) or labels.dtype.kind not in 'iu' or not n or (labels<0).any() or (labels>=5).any():raise ValueError('original-axis integer five-class labels required')
            a=np.asarray(a08.load(i));b=np.asarray(jepa.load(i));check_probabilities(a,n);check_probabilities(b,n)
            self.values.append((torch.as_tensor(a.copy(),dtype=torch.float32,device=device),torch.as_tensor(b.copy(),dtype=torch.float32,device=device),torch.as_tensor(labels.copy(),dtype=torch.int64,device=device)))
    def batch(self,indices):
        if len(indices)!=2:raise ValueError('one native and one mixed whole sequence required')
        i,j=map(int,indices)
        if not 0<=i<len(self.values) or not 0<=j<len(self.values) or self.sequences[i]['kind']!='native' or self.sequences[j]['kind'] not in ['mixed','multi']:raise ValueError('fixed plan must order native then mixed')
        return [self.values[i],self.values[j]]
