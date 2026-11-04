#!/usr/bin/env python3
"""Narrow source-contract bridge and full remaining outer-stage cost admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.proposal_common import *
from scripts.admit_jepa_proposals import main as admit
from echo_routing.temporal.proposal_source_binding import validate

def main():
    verify_freeze('outer_admission_freeze.json');cfg=read(CFG);producer=read(verify_pin(cfg['prior_A59']['producer_identity.json']))
    path='configs/experiments/anchor_edits_v1.json'
    if sha(ROOT/path)!=producer['objects'][path]:raise ValueError('original source config SHA changed')
    old=read(ROOT/path);inputs=read(verify_pin(cfg['prior_A59']['input_manifest.json']));records=[]
    for n in ['cache_fit_manifest.json','cache_outer_manifest.json']:records+=read(verify_pin(cfg['prior_A59'][n]))['records']
    value=validate(cfg['source'],old,producer,inputs,records)
    write(PUB/'source_contract_binding.json',{**value,'original_source_config_sha256':sha(ROOT/path),'old_producer_identity_sha256':cfg['prior_A59']['producer_identity.json']['sha256'],'old_payload_and_axis_qualification_sha256':sha(PUB/'source_cache_qualification.json'),'channel_semantics':'family5_v1 [PLAX,PSAX,A4C,A5C,SC4C] shared A08/J/Base/labels;qualified metadata not shape guess','new_model_forwards':0,'new_optimizer_updates':0})
    admit('outer')
if __name__=='__main__':main()
