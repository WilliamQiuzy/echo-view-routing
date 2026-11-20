# Training

The published model was trained in three stages. Every stage reads EV9V training recordings only.

| Stage | Starts from | Updates | Objective | Input changes | In this package |
|---|---|---|---|---|---|
| 1 | the EchoPrime video encoder | 3,750 × 8 windows | cross-entropy at the marked frame | a gamma of 0.9 on some fragments | `configs/stage1.json` |
| 2 | stage 1 | 3,750 × 8 windows | cross-entropy plus a smoothing penalty between neighbouring windows | as stage 1 | `configs/stage2.json`; its models are also provided |
| 3 | stage 2 | 1,500 × 32 windows | as stage 2 | zoom, shift, rotation, grey maps, gain, gamma, blur, noise, borders | `configs/stage3.json` |

Stage 3 made the published seeds. It is what makes the model usable on another vendor's recordings. On EV9V itself, stage 2 and stage 3 show no consistent difference.

## Stage 3

```bash
# 1. the frame cache: every training and development recording at 10 frames per second, grey (18 GB, once)
echoview-cache --images /data/EV9V/Images --out /data/ev9v_cache

# 2. one seed: 1,500 updates from the stage-2 model of the same seed
echoview-train --config configs/stage3.json --seed 0 --cache /data/ev9v_cache --init stage2 --out runs/C2_s0

# 3. evaluate your three seeds as the published ones are evaluated
pip install -e ".[evaluation]"
mkdir -p mine/C2_s0 && cp runs/C2_s0/final.pt mine/C2_s0/      # and the same for seeds 1 and 2
echoview evaluate --images /data/EV9V/Images --weights mine --unverified --out mine/report.json
```

- **Start weights.** `--init stage2` takes the published stage-2 model of the seed: `training/stage2_s<seed>.pt` in the weights repository on the Hugging Face Hub. It is downloaded and checked against its SHA-256. `--init <file>` starts from a checkpoint of your own, for example a stage-2 model you trained yourself.
- **Hardware.** One run uses 31 GB of GPU memory at batch 32 and took 14 to 18 minutes on an L40S. With `"frames": "host"` the frame cache stays in host memory; with `"gpu"` it is copied to the GPU and needs 16 GB more.
- **Outputs.** `final.pt` holds the last weights. Every published number uses them. `ema.pt` holds their moving average, `train_log.jsonl` the loss and accuracy every 50 updates, `config.json` every option of the run.

### On a machine that can be switched off

A preemptible GPU can disappear in the middle of a run. Two options let stage 3 go on where it stopped.

```bash
echoview-train --config configs/stage3.json --seed 0 --cache /data/ev9v_cache --init stage2 --out runs/C2_s0 \
    --checkpoint-every 100 --resume
```

- `--checkpoint-every 100` writes `checkpoint.pt` into the output folder every 100 updates. The number must be a multiple of `--log-every`.
- `--resume` continues from that file when it is there. Without a checkpoint the run starts at its first update. A folder that holds the finished run is left alone. Give the same command every time the machine comes back.
- **What a checkpoint holds.** The weights, the optimiser's state, the moving average, and the position of every random stream: the windows drawn, the input changes and dropout. The file is about 0.6 GB. It is deleted when the run finishes.
- **How close a continued run comes.** A run that was cut and continued ends where an uncut run ends, up to the difference between two uncut runs. A test cuts a small run on the CPU and measures this: the weights differ by 2e-8 of their length, as two uncut runs do. The same test fails when any part of the checkpoint is left out.
- **Checked on a GPU.** On 2026-11-20 a run of 100 updates of the published recipe (seed 0, without compilation) was killed on an L40S after update 60 and continued from its checkpoint of update 50. Its final weights ended 1.9e-5 of their length away from the uncut run's, which is also the distance between two uncut runs. The logged losses differed by at most 0.0003, against 0.0006 between two uncut runs.
- **Other options are refused.** A checkpoint belongs to one recipe, seed, start file and set of plans. `--resume` with other values stops with a message that names them.

### The recipe

`configs/stage3.json` holds the options of the published seeds.

- **Sampling.** An update holds eight groups of four neighbouring windows, drawn at random from the sampling plans of all three seeds (`src/echoview/data/plans/groups_s*.npz`). The plans hold two kinds of group in equal numbers: groups inside one recording, and groups that cross at least one view change of a stream built from training recordings.
- **Objective.** Cross-entropy at the marked frame, plus 0.15 times a smoothing penalty: the squared difference between the log-probabilities of neighbouring windows of a group, cut at 16 per term. This is the truncated smoothing loss of MS-TCN.
- **Input changes.** Every group gets one random set of changes (`echoview/training/augment.py`). Geometry: zoom, aspect, rotation, shift. Grey levels: a random monotone grey map, gain, gamma, a gain ramp over depth, a slight tint. Degradation: blur or sharpening, speckle-like and additive noise, dark borders.
- **Optimiser.** AdamW, weight decay 1e-4, learning rate 1e-5 for the encoder and 1e-4 for the marker stem and the head, 100 updates of linear warm-up, then a cosine to zero. Gradients are clipped at norm 1. bfloat16 autocast, TF32 and `torch.compile` are on.

### What a rerun reproduces

A rerun draws the same windows as the published run of the same seed. On the same GPU model it also draws the same random input changes; on another GPU the random numbers can differ. It does not give the same weights bit for bit: GPU kernels and compiled code differ between machines and between runs.

How far reruns scatter is known from the project. The three published seeds reach a segment F1@50 of 0.948 to 0.958 on the selection half and 0.978 to 0.986 on the calibration half. The recipe was also repeated nine times with other sampling seeds. Taken three at a time and decoded at one fixed setting, the three-seed means lay between 0.952 and 0.960 on the selection half and between 0.980 and 0.984 on the calibration half.

**Checked against the research trainer.** This package's stage 3 is the research trainer with the options removed that the published recipe never used. On 2026-11-18 both ran 100 updates of seed 0 from the same stage-2 model on one L40S, without compilation. The research trainer ran twice, to show how far two runs of one program differ.

| Compared | Largest loss difference | Accuracy at the ten logged steps | Final weights, difference as a share of their length |
|---|---|---|---|
| research trainer, run 1 and run 2 | 0.0006 | identical | 1.9e-5 |
| this package and research run 1 | 0.0003 | identical | 1.9e-5 |
| this package and research run 2 | 0.0005 | identical | 2.1e-5 |

The package differs from the research trainer no more than the research trainer differs from itself. For scale: the 100 updates move the weights by 7.4e-4 of their length.

**Still open.** A full rerun of the three seeds with this package, evaluated as the published seeds are, has not finished.

## Recipes of your own

`echoview-train` also trains with plain cross-entropy on single windows (`"objective": "ce"`), from a checkpoint or from a pretrained encoder (`--init encoder --encoder <file>`). Every option of `configs/stage3.json` can be given on the command line, which wins over the file. `echoview-train --help` lists them.

## Stages 1 and 2

```bash
# stage 1: from the pretrained encoder
echoview-train --config configs/stage1.json --seed 0 --images /data/EV9V/Images \
    --encoder /path/to/echo_prime_encoder.pt --out runs/stage1_s0

# stage 2: continue the stage-1 model of the same seed
echoview-train --config configs/stage2.json --seed 0 --images /data/EV9V/Images \
    --init runs/stage1_s0/final.pt --out runs/stage2_s0
```

- **The encoder.** Stage 1 starts from the EchoPrime video encoder, `echo_prime_encoder.pt`. Its authors distribute it (github.com/echonet/EchoPrime) under their Academic Software License. It is not redistributed here.
- **The loop.** Update `u` reads the eight windows of row `u` of the seed's sampling plan: `plans/windows_s*.npz` for stage 1, `plans/groups_s*.npz` for stage 2. The frames are decoded from the JPEG files in colour, with the plan's gamma on some fragments. There is no other input change. For the first 375 updates only the marker stem and the head train. Then the encoder joins.
- **Numerics.** Plain float32 with deterministic GPU kernels. One seed fixes the dropout of the network.
- **Hardware.** Eight windows per update need about 13 GB of GPU memory once the encoder trains. A stage took 28 minutes per seed on an L40S during the project.

### What a rerun of stages 1 and 2 reproduces

A rerun feeds the model the same pixels as the published run and follows the same schedule. It does not give the same numbers bit for bit, and two reruns on one machine differ as well: PyTorch has no deterministic GPU kernel for the backward pass of the encoder's 3-D max pooling, and says so in a warning.

This was checked on 2026-11-18 on an L40S with the PyTorch, CUDA and cuDNN versions of the published runs, for seed 0.

- **Inputs.** The sampling plans and the initial weights of all three seeds equal the research files. The research trainer stored a hash of every window's pixels. The frames this package decodes give the same hash for all 90,000 windows of the three stage-1 plans.
- **The first 375 updates**, during which only the marker stem and the head train.

  | Stage | Mean loss, published | Mean loss, two runs of this package | Training accuracy | Largest gradient norm |
  |---|---|---|---|---|
  | 1 | 1.6008545 | 1.6008544 and 1.6008540 | identical | identical to the last digit |
  | 2 | 0.67501289 | 0.67501294 and 0.67501284 | identical | identical to the last digit |

- **Updates 376 to 750**, the first in which the encoder trains.

  | Stage | Mean loss, published | Mean loss, this package | Training accuracy |
  |---|---|---|---|
  | 1 | 0.725491 | 0.725463 | 2,201 and 2,200 of 3,000 windows right |
  | 2 | 0.49662158 | 0.49662125 | identical |

In the first 375 updates the gap to the published log is as large as the gap between two runs of this package.

**Still open.** For updates 376 to 750 two runs of this package have not been compared with each other yet. That comparison was also made before the learning rates of stages 1 and 2 were corrected in their last binary digit (see the changelog), and it will be repeated with the corrected code. Whole reruns of stages 1 and 2 were not made: 750 of 3,750 updates were compared. The pixel hashes of the three stage-2 plans are still being checked.
