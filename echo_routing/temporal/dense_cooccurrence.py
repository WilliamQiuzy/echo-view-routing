"""Fixed DINO patch-channel association descriptor; no anatomical labels implied."""
import numpy as np

def projection():
    a=np.random.default_rng(2026110154).standard_normal((384,32))
    return np.linalg.qr(a)[0]

def descriptor(tokens,p,unpaired=False):
    t=np.asarray(tokens,np.float64)
    if t.ndim!=2 or t.shape[0]!=256 or t.shape[1]!=384 or not np.isfinite(t).all():
        raise ValueError('finite256x384 normalizedpatchtokens required')
    z=t@p;z-=z.mean(0)
    if unpaired:
        shifts=np.random.default_rng(2026110155).choice(256,32,replace=False)
        z=np.column_stack([np.roll(z[:,j],int(s)) for j,s in enumerate(shifts)])
    cov=z.T@z/len(z)
    return np.r_[t.mean(0),cov[np.triu_indices(32)]]
