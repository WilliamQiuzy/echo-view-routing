import numpy as np
import pytest
from echo_routing.temporal.annotation_coverage import geometry, eligible_points, make_plan

def test_geometry_uses_actual_square_source_and_exact_A08_crop():
 from PIL import Image
 from echo_routing.temporal.joint_visual import crop_uint8
 x=np.zeros((500,500,3),np.uint8);x[:,:,0]=np.arange(500)[None,:]%256;x[0,:,1]=255;x[-1,:,2]=255
 f=geometry(x,'F');c=geometry(x,'C');assert f.shape==c.shape==(224,224,3)
 np.testing.assert_array_equal(c,crop_uint8(Image.fromarray(x)))
 assert not np.array_equal(f,c) and f[0,:,1].mean()>0 and c[:,:,1].max()==0

def test_consensus_does_not_guess_offset_or_supervise_unknown():
 records=[{'start':0,'end':30,'true_label':'A4C'},{'start':30,'end':60,'true_label':'A5C'}]
 axis=list(range(60));assert 15 in eligible_points(records,axis,0,30,'A4C',[15]);assert 30 not in eligible_points(records,axis,30,60,'A5C',[45])
 gap=[i+20 if i>=15 else i for i in range(60)];assert 15 not in eligible_points(records,gap,0,30,'A4C',[])
 unknown=[{'start':0,'end':60,'true_label':'NO_VIEW'}];assert eligible_points(unknown,axis,0,60,'A5C',[])==[]

def test_paired_plan_same_parent_labels_and_dense_coverage():
 parents=[{'label':0,'points':list(range(10,20)),'centre':10},{'label':1,'points':list(range(40,50)),'centre':40}]
 p,c=make_plan(parents,updates=20);assert p.shape==c.shape==(20,8);assert set(p.ravel())=={0,1};assert np.array_equal(p,make_plan(parents,20)[0]);assert np.array_equal(c,make_plan(parents,20)[1])
 for ids,centres in zip(p,c):
  assert sum(parents[i]['label'] for i in ids)==4
  for i,x in zip(ids,centres):assert int(x) in parents[int(i)]['points']
 assert len(set(c[p==0]))==10 and len(set(c[p==1]))==10
