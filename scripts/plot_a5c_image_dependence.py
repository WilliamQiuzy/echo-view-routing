#!/usr/bin/env python3
"""Plot public aggregate perturbation maps only; never read private pixels or cases."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results/2026-10-31_a5c_image_dependence'


def main():
    source=json.loads((OUT/'summary.json').read_text())['aggregate_spatial_maps']
    combined={};models=('A08','C','N');groups=('A5C_recovered','A4C_harmed')
    for m in models:
        for g in groups:
            for fill in ('mean','blur'):
                parts=[source[f'{m}_fold{h}_{g}'][fill] for h in (0,1)]
                n=sum(p['valid_anchors'] for p in parts)
                a=sum(np.asarray(p['mean_normalized_absolute_change'])*p['valid_anchors'] for p in parts)/n
                assert a.shape==(49,) and np.isfinite(a).all() and abs(a.sum()-1)<1e-8
                combined[f'{m}_{g}_{fill}']={'anchors':n,'map_7x7':a.reshape(7,7).tolist()}
    fig,axes=plt.subplots(3,4,figsize=(11,8),layout='constrained')
    vmax=max(np.max(v['map_7x7']) for v in combined.values())
    for i,m in enumerate(models):
        for j,(g,fill) in enumerate((g,f) for g in groups for f in ('mean','blur')):
            ax=axes[i,j];v=combined[f'{m}_{g}_{fill}']
            im=ax.imshow(v['map_7x7'],vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(('Recovered A5C' if g==groups[0] else 'Harmed A4C')+' / '+fill,fontsize=11)
            if j==0:ax.set_ylabel(m,fontsize=13)
            ax.text(.02,.98,f'n={v["anchors"]}',transform=ax.transAxes,va='top',color='white',fontsize=9)
    fig.suptitle('Where perturbations change the score',fontsize=16)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Mean per-anchor normalized absolute score change')
    fig.supxlabel('Original 224 × 224 input, 7 × 7 cells. N-defined outcome groups; training-source diagnostic.\nAggregate sensitivity is not anatomical localization or evidence of clinical benefit.',fontsize=10)
    fig.savefig(OUT/'aggregate_sensitivity.png',dpi=180,bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/'aggregate_sensitivity.pdf',bbox_inches='tight',pad_inches=.12)
    (OUT/'aggregate_sensitivity.json').write_text(json.dumps({'method':'Per-anchor absolute perturbation changes divided by their sum, then mean across anchors/folds. Same color scale. No clinical/anatomical interpretation.','maps':combined},indent=2)+'\n')


if __name__=='__main__':main()
