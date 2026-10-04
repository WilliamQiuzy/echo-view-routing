#!/bin/bash
# Geometry pretraining round (configs/experiments/geometry_pretraining_v1.yaml), one GPU job at a time.
# 1. Pretraining: G1 and G2 for seeds 0-2 at the fixed budget, on the seed's bank.
# 2. For each trained encoder: per-frame embeddings (scripts/extract_encoder_features.py), the E1-recipe probe and formal
#    cache (scripts/train_feature_probe.py), then ASRF (M1) and the B0-B3 decoders (simple) with the seed's number.
# Each step writes "<step> <arm>_s<seed> rc=<code>" to logs/geometry_round.log; a failed step stops the round there.
#   bash scripts/geometry_round_run.sh
set -u
cd ~/echo-view-routing
P=runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json
V3=runs/20261002-053334-feature_manifest-6941483d-s0/feature_manifest.json
DOWN=configs/experiments/geometry_downstream_v1.yaml
LOG=logs/geometry_round.log
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 30 >> logs/geometry_round_gpu.csv &   # 20 GB check
GPU_LOGGER=$!; trap 'kill $GPU_LOGGER 2>/dev/null' EXIT
step() {   # step <name> <tag> <command...>
  local name=$1 tag=$2; shift 2
  "$@" > logs/geometry_${name}_${tag}.log 2>&1; local rc=$?
  echo "$(date -Is) ${name} ${tag} rc=${rc}" >> $LOG
  [ $rc -eq 0 ] || { echo "STOPPED at ${name} ${tag}" >> $LOG; exit 1; }
}
for s in 0 1 2; do
  until [ -f data/geometry/banks/seed$s/meta.json ]; do sleep 60; done     # banks render while earlier seeds train
  for arm in G1 G2; do
    step pretrain ${arm}_s$s .venv/bin/python scripts/geometry_pretrain.py --arm $arm --seed $s \
      --bank data/geometry/banks/seed$s --heldout data/geometry/banks/heldout --out runs/geometry/${arm}_s$s
  done
  cmp -s runs/geometry/G1_s$s/batch_digests.txt runs/geometry/G2_s$s/batch_digests.txt \
    && echo "$(date -Is) batches identical s$s" >> $LOG || { echo "STOPPED: G1 and G2 batches differ for seed $s" >> $LOG; exit 1; }
done
for s in 0 1 2; do
  for arm in G1 G2; do
    tag=${arm}_s$s
    step extract $tag .venv/bin/python scripts/extract_encoder_features.py --encoder runs/geometry/$tag/visual_encoder_fp32.pt \
      --protocol $P --manifest data/manifests/ev9v_native.csv --v3-cache cache_v2/features/beb36d80df31 --out cache_v2/geo_raw/$tag
    step probe $tag .venv/bin/python scripts/train_feature_probe.py -c $DOWN --raw-dir cache_v2/geo_raw/$tag --protocol $P --v3-feature-manifest $V3
    MAN=$(grep -o '"run_dir": "[^"]*"' logs/geometry_probe_$tag.log | tail -1 | cut -d'"' -f4)/feature_manifest.json
    [ -f "$MAN" ] || { echo "STOPPED: no feature manifest for $tag" >> $LOG; exit 1; }
    echo "$tag $MAN" >> runs/geometry/manifests.txt
    step downstream $tag .venv/bin/python scripts/model_development_queue.py --config $DOWN --jobs M1,simple --feature-manifest $MAN \
      --protocol $P --seed $s --status runs/geometry/downstream/${tag}_status.json
    .venv/bin/python -c "import json, sys; s = json.load(open('runs/geometry/downstream/${tag}_status.json')); \
sys.exit(0 if all(s[j]['state'] == 'done' for j in ('M1', 'simple')) else 1)" \
      || { echo "STOPPED: a downstream job failed for $tag (the queue exits 0; see its status file)" >> $LOG; exit 1; }
  done
done
echo "$(date -Is) DONE" >> $LOG
