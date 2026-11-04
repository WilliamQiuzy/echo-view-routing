"""Readonly aliases for already pinned public context; never write old namespaces."""
from pathlib import Path,PurePosixPath
from echo_routing.temporal.anchor_cache import sha
from echo_routing.temporal.safety_git import verify as verify_Git
ORIGINALS=(
 'docs/results/2026-11-04_a61_residual_case_analysis/analysis_manifest.json',
 'docs/results/2026-11-04_a61_residual_case_analysis/case_clarification.anonymous.json',
 'docs/results/2026-11-04_a61_residual_case_analysis/case_evidence.anonymous.json',
 'docs/results/2026-11-04_action_safety_ranking/public_source_manifest.json',
)
ARCHIVE_GIT='f2316e23f668f05982ed08e442ed2671463f1a68'


def verify(root,cfg,binding,proofs):
    root=Path(root);expected={p:cfg['prior_source_pins'][p] for p in ORIGINALS};commits={p:ARCHIVE_GIT if 'residual_case_analysis' in p else cfg['prior_scientific_git'] for p in ORIGINALS}
    if set(binding['files'])!=set(ORIGINALS) or binding['source_commits']!=commits or set(proofs)!=set(commits.values()):raise ValueError('exact original context paths and distinct archival/scientific Git provenance required')
    for commit,proof in proofs.items():verify_Git(proof,commit,{p:h for p,h in expected.items() if commits[p]==commit})
    out={};root=root.resolve();prefix=PurePosixPath('docs/results/2026-11-04_anchor_context_risk/readonly_context');folder=root/str(prefix)
    for original,r in binding['files'].items():
        rel=PurePosixPath(r['path'])
        if rel.is_absolute() or any(x in ['.','..'] for x in rel.parts) or rel.parts[:len(prefix.parts)]!=prefix.parts:raise ValueError('canonical new readonly context namespace required')
        p=root
        for part in rel.parts:
            p=p/part
            if p.is_symlink():raise ValueError('readonly context ancestor or leaf symlink forbidden')
        if not p.resolve().is_relative_to(folder) or p.is_symlink() or r['sha256']!=expected[original] or sha(p)!=expected[original]:raise ValueError('relocated public context bytes/basis SHA/new namespace changed')
        out[original]=p
    return out
