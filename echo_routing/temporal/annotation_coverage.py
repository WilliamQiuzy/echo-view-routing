"""A55 exact point-label coverage and full-field versus A08 crop geometry."""
from collections import defaultdict
import numpy as np
from PIL import Image

SEED=2026110155

def geometry(rgb,mode):
 x=np.asarray(rgb)
 if x.dtype!=np.uint8 or x.ndim!=3 or x.shape[-1]!=3 or min(x.shape[:2])<1 or mode not in ('F','C'):
  raise ValueError('uint8RGB and fixedF/C geometry required')
 im=Image.fromarray(x)
 if mode=='F':return np.asarray(im.resize((224,224),Image.Resampling.BILINEAR)).copy()
 w,h=im.size;s=256/min(w,h);im=im.resize((max(224,round(w*s)),max(224,round(h*s))),Image.Resampling.BILINEAR);left=(im.width-224)//2;top=(im.height-224)//2
 return np.asarray(im.crop((left,top,left+224,top+224))).copy()

def eligible_points(records,axis,start,end,label,anchors,excluded=()):
 banned=set(excluded)
 def valid(i):
  if i in banned or not 0<=i<len(axis) or axis[i] is None:return False
  for q in (i,axis[i]):
   for origin in (0,1):
    for inclusive in (0,1):
     if {r['true_label'] for r in records if r['start']-origin<=q<r['end']-origin+inclusive}!={label}:return False
  return True
 lo=max(0,start+1);hi=min(end-1,len(axis));anchors=sorted({int(i) for i in anchors if lo<=i<hi and valid(int(i))});assert len(anchors)<=64
 pool=sorted({i for i in range(lo,hi,3) if valid(i)}|set(anchors));other=[i for i in pool if i not in anchors];n=min(len(other),64-len(anchors));chosen=[] if not n else [other[int(i)] for i in np.linspace(0,len(other)-1,n,dtype=int)]
 return sorted(set(anchors+chosen))

def make_plan(parents,updates=3600):
 y=np.array([p['label'] for p in parents]);rng=np.random.default_rng(SEED);streams=[]
 for label in (0,1):
  pool=np.flatnonzero(y==label)
  if not len(pool) or len(pool)>4*updates:raise ValueError('source exposure cannot cover allparents')
  v=[]
  while len(v)<4*updates:v.extend(rng.permutation(pool).tolist())
  streams.append(np.array(v[:4*updates],np.int64).reshape(updates,4))
 plan=np.concatenate(streams,axis=1);assert set(plan.ravel())==set(range(len(parents)));dense=np.empty_like(plan);visits=np.zeros(len(parents),int);offset=np.random.default_rng(SEED+1).integers(0,2**31,size=len(parents))
 for u,row in enumerate(plan):
  for j,i in enumerate(row):
   points=parents[int(i)]['points']
   if not points:raise ValueError('a fixed parent has no eligible Dcentre')
   dense[u,j]=points[(int(offset[i])+int(visits[i]))%len(points)];visits[i]+=1
 return plan,dense

def coverage(parents,plan,dense):
 old={(tuple(p['key']),int(p['centre'])) for p in parents};pool={(tuple(p['key']),int(c)) for p in parents for c in p['points']};used={(tuple(parents[int(i)]['key']),int(c)) for row,cs in zip(plan,dense) for i,c in zip(row,cs)};labels={tuple(p['key'])+(int(c),):int(p['label']) for p in parents for c in p['points']};patients={p['patient'] for p in parents}
 def counts(points):return {'physical_points':len(points),'not_original_centres':len(points-old),'classes':{str(y):sum(labels[key+(c,)]==y for key,c in points) for y in (0,1)}}
 a,b=counts(pool),counts(used);check=lambda x:x['physical_points']>=2276 and x['not_original_centres']>=1707 and min(x['classes'].values())>=800
 return {'pool':a,'scheduled':b,'patients':len(patients),'quantity_pass':bool(check(a) and check(b) and len(patients)>=20),'original_supervised_centres':len(old),'original_parent_requests':len(parents)}
