window.EXTERNAL_RESULTS = {
  "summary": "A08的3个冻结种子已完成2,470条来源序列。可取得且兼容的原始切面test来自STFM的EV9V；另完成3个外部来源检查。尚无新的独立A4C/A5C cine测试证据。",
  "external": [
    {
      "name": "EchoNet-Dynamic",
      "label": "A4C",
      "n": 1277,
      "mean_recall": 0.9477943095797442,
      "ci95": [
        0.936309057687288,
        0.9582354476637954
      ],
      "correct": [
        1211,
        1210,
        1210
      ],
      "cuts_per_clip": 0.007047768206734534,
      "note": "官方TEST；患者关系未知"
    },
    {
      "name": "EchoNet-LVH",
      "label": "PLAX",
      "n": 343,
      "mean_recall": 0.9572400388726919,
      "ci95": [
        0.9397473275024296,
        0.9727891156462585
      ],
      "correct": [
        317,
        335,
        333
      ],
      "cuts_per_clip": 0.0,
      "note": "官方test；按录像去重"
    },
    {
      "name": "CAMUS",
      "label": "4CH码 → A4C",
      "n": 50,
      "mean_recall": 0.7333333333333334,
      "ci95": [
        0.62,
        0.84
      ],
      "correct": [
        40,
        32,
        38
      ],
      "cuts_per_clip": 0.0,
      "note": "官方testing；半周期0.18–0.51秒；4CH码未经切面复核"
    }
  ],
  "seed_rows": [
    {
      "dataset": "EchoNet-Dynamic",
      "decoder": "固定B3",
      "seed": 0,
      "recall": 0.9483163664839468,
      "cuts_per_clip": 0.007830853563038372
    },
    {
      "dataset": "EchoNet-Dynamic",
      "decoder": "固定B3",
      "seed": 1,
      "recall": 0.9475332811276429,
      "cuts_per_clip": 0.0031323414252153485
    },
    {
      "dataset": "EchoNet-Dynamic",
      "decoder": "固定B3",
      "seed": 2,
      "recall": 0.9475332811276429,
      "cuts_per_clip": 0.010180109631949883
    },
    {
      "dataset": "EchoNet-Dynamic",
      "decoder": "直接argmax",
      "seed": 0,
      "recall": 0.9475332811276429,
      "cuts_per_clip": 0.09945184025058731
    },
    {
      "dataset": "EchoNet-Dynamic",
      "decoder": "直接argmax",
      "seed": 1,
      "recall": 0.9467501957713391,
      "cuts_per_clip": 0.11511354737666406
    },
    {
      "dataset": "EchoNet-Dynamic",
      "decoder": "直接argmax",
      "seed": 2,
      "recall": 0.9451840250587314,
      "cuts_per_clip": 0.09866875489428348
    },
    {
      "dataset": "EchoNet-LVH",
      "decoder": "固定B3",
      "seed": 0,
      "recall": 0.924198250728863,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "EchoNet-LVH",
      "decoder": "固定B3",
      "seed": 1,
      "recall": 0.9766763848396501,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "EchoNet-LVH",
      "decoder": "固定B3",
      "seed": 2,
      "recall": 0.9708454810495627,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "EchoNet-LVH",
      "decoder": "直接argmax",
      "seed": 0,
      "recall": 0.9183673469387755,
      "cuts_per_clip": 0.11078717201166181
    },
    {
      "dataset": "EchoNet-LVH",
      "decoder": "直接argmax",
      "seed": 1,
      "recall": 0.9708454810495627,
      "cuts_per_clip": 0.11661807580174927
    },
    {
      "dataset": "EchoNet-LVH",
      "decoder": "直接argmax",
      "seed": 2,
      "recall": 0.9766763848396501,
      "cuts_per_clip": 0.09620991253644315
    },
    {
      "dataset": "CAMUS",
      "decoder": "固定B3",
      "seed": 0,
      "recall": 0.8,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "CAMUS",
      "decoder": "固定B3",
      "seed": 1,
      "recall": 0.64,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "CAMUS",
      "decoder": "固定B3",
      "seed": 2,
      "recall": 0.76,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "CAMUS",
      "decoder": "直接argmax",
      "seed": 0,
      "recall": 0.8,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "CAMUS",
      "decoder": "直接argmax",
      "seed": 1,
      "recall": 0.64,
      "cuts_per_clip": 0.0
    },
    {
      "dataset": "CAMUS",
      "decoder": "直接argmax",
      "seed": 2,
      "recall": 0.76,
      "cuts_per_clip": 0.0
    }
  ],
  "original": [
    {
      "name": "视频Transformer＋固定B3（A08）",
      "ours": true,
      "seed": "0 / 1 / 2 均值",
      "n": 800,
      "accuracy": 0.9804166666666667,
      "macro_f1": 0.9537008163877992,
      "macro_ci95": [
        0.927289354776908,
        0.9729133148774649
      ],
      "a4c": 0.9509379509379509,
      "a5c": 0.9166666666666666
    },
    {
      "name": "A08＋固定B3",
      "ours": true,
      "seed": "0",
      "n": 800,
      "accuracy": 0.9775,
      "macro_f1": 0.9484549375583384,
      "macro_ci95": [
        0.9204902689610867,
        0.9691976367302841
      ],
      "a4c": 0.9393939393939394,
      "a5c": 0.925
    },
    {
      "name": "A08＋固定B3",
      "ours": true,
      "seed": "1",
      "n": 800,
      "accuracy": 0.98125,
      "macro_f1": 0.9541104574519246,
      "macro_ci95": [
        0.9272087958004457,
        0.9742288802006409
      ],
      "a4c": 0.961038961038961,
      "a5c": 0.875
    },
    {
      "name": "A08＋固定B3",
      "ours": true,
      "seed": "2",
      "n": 800,
      "accuracy": 0.9825,
      "macro_f1": 0.9585370541531347,
      "macro_ci95": [
        0.9317336693702476,
        0.9778512360780716
      ],
      "a4c": 0.9523809523809523,
      "a5c": 0.95
    },
    {
      "name": "STFM 原生规则（旧冻结权重）",
      "ours": false,
      "seed": "100",
      "n": 800,
      "accuracy": 0.97,
      "macro_f1": 0.9653393303239074,
      "macro_ci95": [
        0.9443289195031562,
        0.9817716679757521
      ],
      "a4c": 0.974025974025974,
      "a5c": 0.95
    },
    {
      "name": "STFM 先验加一次（同一旧权重）",
      "ours": false,
      "seed": "100",
      "n": 800,
      "accuracy": 0.96875,
      "macro_f1": 0.9648485905168421,
      "macro_ci95": [
        0.9438606220276203,
        0.9810514346625023
      ],
      "a4c": 0.974025974025974,
      "a5c": 0.95
    }
  ],
  "original_details": [
    {
      "name": "STFM 原生规则（旧冻结权重）",
      "ours": false,
      "population": "九码 / 888",
      "accuracy": 0.9448198198198198,
      "macro_f1": 0.9093818329259263
    },
    {
      "name": "STFM 先验加一次（同一旧权重）",
      "ours": false,
      "population": "九码 / 888",
      "accuracy": 0.9448198198198198,
      "macro_f1": 0.9093818329259263
    },
    {
      "name": "A08 直接argmax，s0",
      "ours": true,
      "population": "五类 / 800",
      "accuracy": 0.98,
      "macro_f1": 0.9537862999628717
    },
    {
      "name": "A08 直接argmax，s1",
      "ours": true,
      "population": "五类 / 800",
      "accuracy": 0.98375,
      "macro_f1": 0.9585063893140198
    },
    {
      "name": "A08 直接argmax，s2",
      "ours": true,
      "population": "五类 / 800",
      "accuracy": 0.9825,
      "macro_f1": 0.9585370541531347
    }
  ],
  "history": [
    {
      "name": "A08＋B3（s0/s2均值）",
      "ours": true,
      "native": 0.9534959958557365,
      "pairs": 0.9784866697488057,
      "multi": 0.918918918918919,
      "fp": 41.0,
      "fp_short": 20.0,
      "fn": 40.0,
      "fn_short": 15.5
    },
    {
      "name": "EchoViewCLIP stage1",
      "ours": false,
      "native": 0.9719364414178454,
      "pairs": 0.9889196675900277,
      "multi": 0.8667305848513902,
      "fp": 92.0,
      "fp_short": 62.0,
      "fn": 47.0,
      "fn_short": 11.0
    },
    {
      "name": "STFM 数值帧序 s100",
      "ours": false,
      "native": 0.9596268824889631,
      "pairs": 0.970954356846473,
      "multi": 0.859086491739553,
      "fp": 88.0,
      "fp_short": 60.0,
      "fn": 57.0,
      "fn_short": 21.0
    },
    {
      "name": "ASFormer（EV9V适应）",
      "ours": false,
      "native": 0.9558519059280697,
      "pairs": 0.9807162534435262,
      "multi": 0.8563106796116505,
      "fp": 90.0,
      "fp_short": 48.0,
      "fn": 58.0,
      "fn_short": 16.0
    },
    {
      "name": "EchoPrime＋B3",
      "ours": false,
      "native": 0.943459031209738,
      "pairs": 0.9875862068965517,
      "multi": 0.8510204081632653,
      "fp": 64.0,
      "fp_short": 17.0,
      "fn": 82.0,
      "fn_short": 23.0
    },
    {
      "name": "EchoPrime 逐帧",
      "ours": false,
      "native": 0.943459031209738,
      "pairs": 0.7427341227125942,
      "multi": 0.4957678355501814,
      "fp": 745.0,
      "fp_short": 696.0,
      "fn": 89.0,
      "fn_short": 15.0
    }
  ],
  "availability": {
    "audited_utc": "2026-10-30",
    "scope": "External methods used as comparators or representation sources in this project; original pixels plus independent target truth are required.",
    "models": [
      {
        "model": "STFM",
        "original_test": "EV9V official test: 888 cines / nine codes",
        "status": "executable_shared_labels",
        "pixels": true,
        "truth": true,
        "compatible_endpoint": "800 native cines mapped to five families; full nine-code endpoint baseline-only",
        "source": "https://huggingface.co/datasets/bgx666/EV9V/tree/main",
        "code": "https://github.com/bgx666/stfm",
        "limit": "Same EV9V source and historically inspected test, not independent external confirmation. New official 61ea9b6 is distinct from project-pinned 532f60b."
      },
      {
        "model": "EchoViewCLIP",
        "original_test": "Guangdong hospital: 4625 standard-view test cines / 38 classes; 200 OOD test",
        "status": "original_pixels_and_truth_unavailable",
        "pixels": false,
        "truth": false,
        "compatible_endpoint": null,
        "source": "https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf",
        "code": "https://github.com/xmed-lab/EchoViewCLIP",
        "limit": "Official repository has labels/template splits, not the original cine test bundle. CAMUS quality experiment is a different task."
      },
      {
        "model": "EchoPrime",
        "original_test": "4000 internal test videos / 58 views; institutional clinical cohorts",
        "status": "institutional_DUA_and_task_mismatch",
        "pixels": false,
        "truth": false,
        "compatible_endpoint": null,
        "source": "https://www.nature.com/articles/s41586-025-09850-x",
        "code": "https://github.com/echonet/EchoPrime",
        "limit": "Original expert 58-view truth unavailable; released project classifier has 11 coarse outputs. Public MIMIC pixels do not supply original expert view truth."
      },
      {
        "model": "ASFormer",
        "original_test": "GTEA / 50Salads / Breakfast action segmentation",
        "status": "available_but_incompatible",
        "pixels": false,
        "truth": true,
        "compatible_endpoint": null,
        "source": "https://zenodo.org/records/3625992",
        "code": "https://github.com/ChinaYi/ASFormer",
        "limit": "Released original package contains I3D features and action labels; frozen cardiac five-class model cannot consume/predict that contract."
      },
      {
        "model": "MS-TCN",
        "original_test": "GTEA / 50Salads / Breakfast action segmentation",
        "status": "available_but_incompatible",
        "pixels": false,
        "truth": true,
        "compatible_endpoint": null,
        "source": "https://zenodo.org/records/3625992",
        "code": "https://github.com/yabufarha/ms-tcn",
        "limit": "Feature/action original benchmark is not a cardiac cine recognition test."
      },
      {
        "model": "ASRF",
        "original_test": "GTEA / 50Salads / Breakfast action segmentation",
        "status": "available_but_incompatible",
        "pixels": false,
        "truth": true,
        "compatible_endpoint": null,
        "source": "https://zenodo.org/records/3625992",
        "code": "https://github.com/yiskw713/asrf",
        "limit": "Action boundaries and labels are incompatible with frozen A08 outputs."
      },
      {
        "model": "FACT",
        "original_test": "Breakfast / GTEA / EgoProcel / Epic-Kitchens",
        "status": "available_but_incompatible",
        "pixels": false,
        "truth": true,
        "compatible_endpoint": null,
        "source": "https://github.com/ZijiaLewisLu/CVPR2024-FACT",
        "limit": "Action feature/label task; official repository also distinguishes replacement replication weights from lost original weights."
      },
      {
        "model": "EchoFine / TTE47",
        "original_test": "5447 static test images / three expert annotations / 47 views",
        "status": "access_form_and_static_input",
        "pixels": false,
        "truth": false,
        "compatible_endpoint": null,
        "source": "https://www.thrive-centre.com/datasets/TTE47",
        "code": "https://github.com/PreshenNaidoo/EchoFine",
        "limit": "No directly obtainable complete pixel/annotation test package verified. Static images cannot be repeated into an invented 16-frame cine test."
      },
      {
        "model": "EchoJEPA / V-JEPA",
        "original_test": "Private Toronto/Chicago view test; public EchoNet LVEF tests",
        "status": "original_view_test_unavailable",
        "pixels": false,
        "truth": false,
        "compatible_endpoint": "Auxiliary A4C source check on EchoNet-Dynamic; not comparable with LVEF MAE",
        "source": "https://arxiv.org/html/2602.02603v2",
        "code": "https://github.com/bowang-lab/EchoJEPA",
        "limit": "Paper view task says 12 classes; released view configuration says 13 and references private UHN CSVs. No original view-test bundle."
      },
      {
        "model": "PanEcho",
        "original_test": "Private clinical study test; public EchoNet-Dynamic / EchoNet-LVH measurement endpoints",
        "status": "public_data_different_target",
        "pixels": true,
        "truth": true,
        "compatible_endpoint": "Auxiliary A4C / PLAX source checks, not clinical or measurement endpoint comparison",
        "source": "https://jamanetwork.com/journals/jama/fullarticle/2835630",
        "code": "https://github.com/CarDS-Yale/PanEcho",
        "limit": "Original tasks are clinical interpretation / regression. Frozen A08 has no EF or wall-thickness output."
      },
      {
        "model": "AoVdetector",
        "original_test": "PLAX / PSAX / 3CH aortic-valve box detection",
        "status": "original_test_unavailable_and_target_mismatch",
        "pixels": false,
        "truth": false,
        "compatible_endpoint": null,
        "source": "https://github.com/CardiovascularImagingVallHebron/AoVdetector",
        "code": "https://huggingface.co/CardiovascularImagingVHIR/AoVdetector",
        "limit": "Code / four-row example / weights do not provide original full test pixels, annotations or split. A08 predicts views, not boxes."
      },
      {
        "model": "EchoSlicer / Echo-POSED",
        "original_test": "3D geometry and classifier-guided standard-plane slices",
        "status": "incompatible_or_exact_test_unverified",
        "pixels": true,
        "truth": false,
        "compatible_endpoint": null,
        "source": "https://github.com/echonet/3d-echo",
        "limit": "Classifier-guided planes are not independent true A4C/A5C cine labels."
      },
      {
        "model": "EchoFM / Echo-Vision-FM (intake only)",
        "original_test": "TMED-2 still images, CAMUS segmentation, EchoNet regression and private cine tasks",
        "status": "public_data_different_target",
        "pixels": true,
        "truth": true,
        "compatible_endpoint": "Auxiliary CAMUS 4CH-coded source check; not segmentation Dice or quality accuracy",
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12616925/",
        "code": "https://github.com/SekeunKim/EchoFM",
        "limit": "Still-image, segmentation and regression scores cannot be compared with cardiac view accuracy. No new independent A5C cine test verified."
      }
    ],
    "additional_resources": [
      {
        "name": "EchoXFlow",
        "source": "https://huggingface.co/datasets/Ahus-AIM/EchoXFlow",
        "disposition": "Public cine pixels, but released view labels are model predictions; not independent expert test truth."
      },
      {
        "name": "Unity Imaging",
        "source": "https://data.unityimaging.net/",
        "disposition": "Sparse still frames, evaluation-only project restriction; no dense cine test. Potential EchoFine source overlap unverified."
      },
      {
        "name": "MedSapiens / EchoNet Measurements",
        "source": "https://github.com/echonet/measurements",
        "disposition": "Landmarks/measurement targets; no directly matched A4C/A5C view endpoint."
      },
      {
        "name": "Pediatric",
        "source": "https://echonet.github.io/pediatric/",
        "disposition": "Existing project permissions/history restrictions retained; not a new untouched test and not included in this run."
      }
    ]
  },
  "paired_original": [
    {
      "name": "STFM 原生规则",
      "n_cases": 800,
      "groups": 125,
      "grouping": "paired sessions; identical case membership",
      "accuracy_difference": 0.01041666666666674,
      "accuracy_difference_ci95": [
        -0.003162784031686394,
        0.02504975957647753
      ],
      "macro_f1_difference": -0.011638513936108152,
      "macro_f1_difference_ci95": [
        -0.031128556970807696,
        0.006529581212555601
      ]
    },
    {
      "name": "STFM 先验加一次（同旧权重）",
      "n_cases": 800,
      "groups": 125,
      "grouping": "paired sessions; identical case membership",
      "accuracy_difference": 0.011666666666666714,
      "accuracy_difference_ci95": [
        -0.0021355626070874213,
        0.026810520939734326
      ],
      "macro_f1_difference": -0.011147774129042887,
      "macro_f1_difference_ci95": [
        -0.03071250312010729,
        0.006890299335647024
      ]
    }
  ],
  "resources": {
    "terminal_receipts": {
      "external_original_test_v1.log": {
        "sha256": "de1b11160d9c3fbfe0d8b245ddccea5398f6c42168f27a91e880ac5c37394d39",
        "jobs": [
          {
            "command": "\".venv/bin/python -u scripts/external_original_tests.py --stage run\"",
            "user_seconds": 960.83,
            "system_seconds": 28.19,
            "elapsed": "17:02.36",
            "exit_status": 1
          }
        ]
      },
      "external_original_test_v1_continuation.log": {
        "sha256": "5c7362df403c49ca4c4dda0db67e9fbb495177c21826ad3c1b57158a144fb463",
        "jobs": [
          {
            "command": "\".venv/bin/python -u scripts/external_original_tests.py --stage run --datasets lvh camus\"",
            "user_seconds": 313.83,
            "system_seconds": 45.65,
            "elapsed": "4:36.81",
            "exit_status": 0
          },
          {
            "command": "\".venv/bin/python -u scripts/external_native_baselines.py --stage run\"",
            "user_seconds": 408.07,
            "system_seconds": 24.73,
            "elapsed": "2:04.73",
            "exit_status": 0
          }
        ]
      },
      "external_original_test_v1_audit_attempt1.log": {
        "sha256": "168f1ced9cdd4f6095da329d7ab9f7fd0b6fa6b1698615c52949272f70ba208d",
        "jobs": [
          {
            "command": "\".venv/bin/python -u scripts/audit_external_original_tests.py\"",
            "user_seconds": 0.69,
            "system_seconds": 0.01,
            "elapsed": "0:00.13",
            "exit_status": 1
          }
        ]
      },
      "external_original_test_v1_audit_attempt2.log": {
        "sha256": "74811dee2ad9ea97cadc640cb4b4f1bbf7af06b9ecec27e286d94688ada99ab3",
        "jobs": [
          {
            "command": "\".venv/bin/python -u scripts/audit_external_original_tests.py\"",
            "user_seconds": 12.16,
            "system_seconds": 1.11,
            "elapsed": "0:11.46",
            "exit_status": 1
          }
        ]
      },
      "external_original_test_v1_audit_attempt3.log": {
        "sha256": "798fe75b3fe2b680219068a8306e1cef6bb42ab3f4582559d51e94e8458e8135",
        "jobs": [
          {
            "command": "\".venv/bin/python -u scripts/audit_external_original_tests.py\"",
            "user_seconds": 19.03,
            "system_seconds": 1.18,
            "elapsed": "0:12.98",
            "exit_status": 0
          }
        ]
      },
      "external_original_test_v1_audit.log": {
        "sha256": "74cb91725c609fd9e5f2a55f6287a686ede853399338347b5e92de23f4f4aa81",
        "jobs": [
          {
            "command": "\".venv/bin/python scripts/audit_external_original_tests.py\"",
            "user_seconds": 19.26,
            "system_seconds": 1.09,
            "elapsed": "0:13.48",
            "exit_status": 0
          }
        ]
      }
    },
    "stages": [
      {
        "stage": "prepare",
        "ended_utc": "2026-10-30T03:59:51Z",
        "wall_seconds": 0.8981783390045166,
        "process_cpu_seconds": 0.8044329909999997
      },
      {
        "stage": "check",
        "ended_utc": "2026-10-30T04:00:12Z",
        "wall_seconds": 7.218328237533569,
        "process_cpu_seconds": 9.001459667999999
      },
      {
        "stage": "check",
        "ended_utc": "2026-10-30T04:22:28Z",
        "wall_seconds": 7.081995487213135,
        "process_cpu_seconds": 8.811783634000001
      },
      {
        "stage": "run",
        "ended_utc": "2026-10-30T04:27:53Z",
        "wall_seconds": 273.60732531547546,
        "process_cpu_seconds": 355.359114392,
        "gpu_forward_seconds": 104.09862565994263,
        "windows": 7689,
        "datasets": [
          "lvh",
          "camus"
        ]
      },
      {
        "stage": "STFM-original-native",
        "ended_utc": "2026-10-30T04:29:58Z",
        "wall_seconds": 121.5306875705719,
        "process_cpu_seconds": 428.71685260000004,
        "gpu_forward_seconds": 4.982695817947388
      }
    ],
    "run_storage_bytes": 11789234,
    "created_utc": "2026-10-30",
    "gpu_model": "NVIDIA L40S",
    "training_runs": 0,
    "a08_seed_runs": 3,
    "source_sequences": 2470,
    "baseline_sequences": 888,
    "gpu_forward_seconds": 875.7201302051544,
    "gpu_job_elapsed_seconds": 1423.9,
    "input_qualification_seconds": {
      "LVH_all_PTS": 80.6620728969574,
      "CAMUS_all_arrays": 4.91621994972229
    },
    "accounting_limit": "GPU job wall includes loading, decoding and CPU work; measured forward is not GPU rental time. Process CPU receipts are not total agent/development/source-audit CPU; no complete dollar or cloud reservation charge is claimed. No new dataset download was required; existing server data were reused. Protocol/check wall is separate. Initial failed LVH job and all four audit attempts are retained. Storage is file bytes for this run, excluding shared existing data/models and local/hosting copies."
  },
  "model_identity": "A08全相邻平滑视频Transformer＋冻结B3；s0/s1/s2；权重SHA前缀：9b144925f32f, c8b09ecaf286, f95fed7ba78c",
  "verification_note": "逐例缓存已独立重建概率、完整标签序列、计数和指标；开发集输出复现、原始帧率/窗口/标记核查通过。95%区间为固定模型下分组重采样的描述性区间；患者独立性和上游预训练重叠仍有限制。",
  "verification": {
    "passed": true,
    "created_utc": "2026-10-30T04:40:31Z",
    "checks": {
      "frozen_protocol_config_code_weights": true,
      "source_metadata_hashes": true,
      "official_membership_and_complete_truth": true,
      "exact_cache_counts_keys_identities": true,
      "finite_logits_shapes_centres_sampling_timing": true,
      "complete_independently_decoded_labels": true,
      "independent_native_majority_confusion_metrics_cuts": true,
      "STFM_raw9_shared5_both_fusions": true,
      "unsupported_predictions_retained_as_errors": true,
      "PSAX_probability_mass_grouped": true,
      "CAMUS_whole_source_CPU_audit_reference": true,
      "grouped_bootstrap_same_seed_membership": true,
      "prospective_LVH_timing_repair": {
        "qualified_recordings": 343,
        "max_pts_delta_error_s": 1.349614864309956e-15,
        "model_weights_B3_preprocess_windows_membership_unchanged": true
      },
      "dataset_protocol_bindings": {
        "ev9v": "protocol_v1.json",
        "dynamic": "protocol_v1.json",
        "lvh": "protocol.json",
        "camus": "protocol.json"
      },
      "A26_exact_native_recount": {
        "s0": {
          "accuracy": 0.9775,
          "macro_f1": 0.9484549375583384,
          "exact": true
        },
        "s2": {
          "accuracy": 0.9825,
          "macro_f1": 0.9585370541531347,
          "exact": true
        }
      }
    },
    "counts": {
      "ev9v": 800,
      "dynamic": 1277,
      "lvh": 343,
      "camus": 50,
      "ev9v_sessions": 125,
      "STFM_original_raw9": 888,
      "STFM_shared5": 800,
      "native_cache_files": 2470,
      "native_10Hz_samples_per_seed": 103122,
      "model_seeds": [
        0,
        1,
        2
      ],
      "bootstrap_draws": 2000
    },
    "hashes": {
      "preparation.json": "9cfe576a90e88702e2b2750d50bbd1f102f39a9dcde1136c245c8b3f62b66c63",
      "preparation_v2.json": "08a34797b505303d32fd9171df1edbb34157f3c5b749e09853254a3af00f298d",
      "baseline_preparation.json": "fdc75c818a7a5247bde073b70fc9414b75ffddfd3361f2dbd9b81addcbfd1a1f",
      "input_audit.json": "322133d4c220f5ea53ef82ddbd92e4325e6286e0a1bf0b2e9905e989e1dd4bc2",
      "timing_amendment.json": "2b78be70ed67f0ab239ba50a9026571176aa11b998a789608d50af63dadb54ef",
      "checks_v1.json": "4d3125fd87817752c27c81675e482cc1e71c6247778912c7135b076501818105",
      "checks.json": "9ad85f53ef65fa1c62de796e03b1335521302d3451f1b7b867f47ffc080cd727",
      "historical_A26": "e7702bc66b3b2bc018263a9d60bad5b382c1d197124d15e90c19aa1b95d34a8f",
      "choices": "b2a731e5f4393187ee9954f710821ac04938652f0a7d00e0381ad0b8b21cf27e",
      "protocol": "08a34797b505303d32fd9171df1edbb34157f3c5b749e09853254a3af00f298d",
      "protocol_v1": "9cfe576a90e88702e2b2750d50bbd1f102f39a9dcde1136c245c8b3f62b66c63",
      "protocol_bindings": "e937041c5ac8d5c0bac292783381f5cf251719602d57f14b3c1ca31ab478416a",
      "baseline_protocol": "fdc75c818a7a5247bde073b70fc9414b75ffddfd3361f2dbd9b81addcbfd1a1f",
      "config": "66c0b8ed020761080f231a0d1fd6f81b42d8e7f709e7b73df4d3fa5549500e37",
      "manifest": "d719bbbc2648f16b97cd93fc679f6fdc423d5928fe71a9e84d02727f194af0bc",
      "manifest_v1": "3cef0b62e336a08b7bc7813fa9517a26784338c45dc4a7899a07f83f1f119215",
      "baseline_manifest": "bb90accb5450df084a86cdb127991dc0c3a9b5eba017fcf343d9508f32f5bb96",
      "native_code": {
        "scripts/external_original_tests.py": "c127ff446acfaf25186b1482cd1a1acd6bcab073c5a33a741198697bbfa71186",
        "echo_routing/evaluate/external_native.py": "140b576d3cecb4b17a33a8c9c9111268fa098ba23efdeaf2280d81dda438013b",
        "echo_routing/vit/target_mvit.py": "6b8199420fd813571820e441c0f16b1c75ae9cb889a656b527d747c182afecf8",
        "echo_routing/temporal/joint_visual.py": "6aa655fda17b7f24c948183bc7bd20a075fdd01c7966a59922a3ffea47a6f048",
        "scripts/amend_external_lvh_timing.py": "6aa82894a90cda1c4cbb6390b3611e59b8fd686d0b2f03b6fd9ec4dde856b557"
      },
      "baseline_code": "ffc0038f807ef78d78199f2beffdba1a0007d6ef24f7a372b3e1f3c5d84d52fe",
      "baseline_mapping": "48e6674b556304c2c91b3ecc1bdadaa7ed42aaf98e656ed2474b59b9f2b14de1",
      "baseline_weights": "c09b9f7a451a45d6b77e00b7e5b2f8fde5cece206d84e9fc0b2b859f30e08aa2",
      "native_weights": {
        "0": "9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc",
        "1": "c8b09ecaf286c7b66fc0eec69fdd102f98930aaa657a7604e3204c3845a8e172",
        "2": "f95fed7ba78cc0b5e2474c71a156356507b4588a4b324957fd5b53fe66ee8472"
      },
      "LVH_timing_qualification": "3b8457ab4208d9c7e6fc162c1890101a71188fcb4fa809d26abfd7d8130a2ae3",
      "source_metadata": {
        "file_count": 54,
        "aggregate_sha256": "64a60d8b17f0aff617fd651dfe830c9094934b66ca628ca328cec917740fbce6"
      },
      "native_prediction_aggregate_sha256": "7464902152a5c8abb8868739ad4e7f6d7620760c19f3508fb1cb56667d57fbde",
      "baseline_prediction_aggregate_sha256": "8cd359bbdd7150e2e7aa26a7a21479e0cf911a1867eb00d9bcdf1d28ff94dcb4",
      "native_results": "fa5b7f7ad05d8716f783586c412a5b7696ecdc1a3ac513c7e062ea1d5b3c7ae9",
      "stfm_native_results": "4e9ac80851b6a9a6fbff432a984ffbb9b20c5326bb75bb7250370f8d0ffe8c21",
      "audit_script": "9d80863b95e22d4d9341902a0b757f298c6fa284fae85fe48f356e31f6586ed8",
      "uncertainty": "5af98b90fd3ecd64e90636c56ef028ae065b09b6f2acdb5d359f713124fe82bb"
    },
    "recounted_native_metrics": {
      "ev9v": {
        "argmax_s0": {
          "n_clips": 800,
          "accuracy": 0.98,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.9437229437229437,
            "A5C": 0.95,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9537862999628717,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              219,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              218,
              11,
              2,
              0
            ],
            [
              0,
              0,
              2,
              38,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 88,
          "internal_cuts_per_clip": 0.11,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 1.141703906587862,
          "n_samples": 46507
        },
        "argmax_s1": {
          "n_clips": 800,
          "accuracy": 0.98375,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.974025974025974,
            "A5C": 0.85,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9585063893140198,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              219,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              225,
              4,
              2,
              0
            ],
            [
              0,
              0,
              6,
              34,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 74,
          "internal_cuts_per_clip": 0.0925,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 0.9600691941761568,
          "n_samples": 46507
        },
        "argmax_s2": {
          "n_clips": 800,
          "accuracy": 0.9825,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.9523809523809523,
            "A5C": 0.95,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9585370541531347,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              219,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              220,
              9,
              2,
              0
            ],
            [
              0,
              0,
              2,
              38,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 97,
          "internal_cuts_per_clip": 0.12125,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 1.25846907885253,
          "n_samples": 46507
        },
        "frozen_B3_s0": {
          "n_clips": 800,
          "accuracy": 0.9775,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.9393939393939394,
            "A5C": 0.925,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9484549375583384,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              219,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              217,
              12,
              2,
              0
            ],
            [
              0,
              0,
              3,
              37,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 25,
          "internal_cuts_per_clip": 0.03125,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 0.3243477007351881,
          "n_samples": 46507
        },
        "frozen_B3_s1": {
          "n_clips": 800,
          "accuracy": 0.98125,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.961038961038961,
            "A5C": 0.875,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9541104574519246,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              219,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              222,
              7,
              2,
              0
            ],
            [
              0,
              0,
              5,
              35,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 20,
          "internal_cuts_per_clip": 0.025,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 0.2594781605881505,
          "n_samples": 46507
        },
        "frozen_B3_s2": {
          "n_clips": 800,
          "accuracy": 0.9825,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.9523809523809523,
            "A5C": 0.95,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9585370541531347,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              219,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              220,
              9,
              2,
              0
            ],
            [
              0,
              0,
              2,
              38,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 23,
          "internal_cuts_per_clip": 0.02875,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 0.2983998846763731,
          "n_samples": 46507
        }
      },
      "dynamic": {
        "argmax_s0": {
          "n_clips": 1277,
          "accuracy": 0.9475332811276429,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 1277,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9475332811276429,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              1,
              1210,
              64,
              2,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 127,
          "internal_cuts_per_clip": 0.09945184025058731,
          "duration_s": 4403.130137983133,
          "internal_cuts_per_minute": 1.7305870508497767,
          "n_samples": 44537
        },
        "argmax_s1": {
          "n_clips": 1277,
          "accuracy": 0.9467501957713391,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 1277,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9467501957713391,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              2,
              1209,
              63,
              3,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 147,
          "internal_cuts_per_clip": 0.11511354737666406,
          "duration_s": 4403.130137983133,
          "internal_cuts_per_minute": 2.003120444684387,
          "n_samples": 44537
        },
        "argmax_s2": {
          "n_clips": 1277,
          "accuracy": 0.9451840250587314,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 1277,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9451840250587314,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              1,
              1207,
              61,
              8,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 126,
          "internal_cuts_per_clip": 0.09866875489428348,
          "duration_s": 4403.130137983133,
          "internal_cuts_per_minute": 1.716960381158046,
          "n_samples": 44537
        },
        "frozen_B3_s0": {
          "n_clips": 1277,
          "accuracy": 0.9483163664839468,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 1277,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9483163664839468,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              1,
              1211,
              62,
              3,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 10,
          "internal_cuts_per_clip": 0.007830853563038372,
          "duration_s": 4403.130137983133,
          "internal_cuts_per_minute": 0.13626669691730525,
          "n_samples": 44537
        },
        "frozen_B3_s1": {
          "n_clips": 1277,
          "accuracy": 0.9475332811276429,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 1277,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9475332811276429,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              2,
              1210,
              63,
              2,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 4,
          "internal_cuts_per_clip": 0.0031323414252153485,
          "duration_s": 4403.130137983133,
          "internal_cuts_per_minute": 0.0545066787669221,
          "n_samples": 44537
        },
        "frozen_B3_s2": {
          "n_clips": 1277,
          "accuracy": 0.9475332811276429,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 1277,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9475332811276429,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              1210,
              59,
              8,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 13,
          "internal_cuts_per_clip": 0.010180109631949883,
          "duration_s": 4403.130137983133,
          "internal_cuts_per_minute": 0.17714670599249682,
          "n_samples": 44537
        }
      },
      "lvh": {
        "argmax_s0": {
          "n_clips": 343,
          "accuracy": 0.9183673469387755,
          "class_support": {
            "PLAX": 343,
            "PSAX": 0,
            "A4C": 0,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": 0.9183673469387755,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              315,
              4,
              9,
              14,
              1,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 38,
          "internal_cuts_per_clip": 0.11078717201166181,
          "duration_s": 1174.7972042670065,
          "internal_cuts_per_minute": 1.9407604918693733,
          "n_samples": 11884
        },
        "argmax_s1": {
          "n_clips": 343,
          "accuracy": 0.9708454810495627,
          "class_support": {
            "PLAX": 343,
            "PSAX": 0,
            "A4C": 0,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": 0.9708454810495627,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              333,
              2,
              5,
              2,
              1,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 40,
          "internal_cuts_per_clip": 0.11661807580174927,
          "duration_s": 1174.7972042670065,
          "internal_cuts_per_minute": 2.0429057809151296,
          "n_samples": 11884
        },
        "argmax_s2": {
          "n_clips": 343,
          "accuracy": 0.9766763848396501,
          "class_support": {
            "PLAX": 343,
            "PSAX": 0,
            "A4C": 0,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": 0.9766763848396501,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              335,
              2,
              5,
              1,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 33,
          "internal_cuts_per_clip": 0.09620991253644315,
          "duration_s": 1174.7972042670065,
          "internal_cuts_per_minute": 1.6853972692549821,
          "n_samples": 11884
        },
        "frozen_B3_s0": {
          "n_clips": 343,
          "accuracy": 0.924198250728863,
          "class_support": {
            "PLAX": 343,
            "PSAX": 0,
            "A4C": 0,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": 0.924198250728863,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              317,
              3,
              9,
              13,
              1,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 1174.7972042670065,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 11884
        },
        "frozen_B3_s1": {
          "n_clips": 343,
          "accuracy": 0.9766763848396501,
          "class_support": {
            "PLAX": 343,
            "PSAX": 0,
            "A4C": 0,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": 0.9766763848396501,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              335,
              2,
              4,
              1,
              1,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 1174.7972042670065,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 11884
        },
        "frozen_B3_s2": {
          "n_clips": 343,
          "accuracy": 0.9708454810495627,
          "class_support": {
            "PLAX": 343,
            "PSAX": 0,
            "A4C": 0,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": 0.9708454810495627,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              333,
              1,
              8,
              1,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 1174.7972042670065,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 11884
        }
      },
      "camus": {
        "argmax_s0": {
          "n_clips": 50,
          "accuracy": 0.8,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 50,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.8,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              5,
              40,
              5,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 17.586281364234193,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 194
        },
        "argmax_s1": {
          "n_clips": 50,
          "accuracy": 0.64,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 50,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.64,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              1,
              32,
              13,
              4,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 17.586281364234193,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 194
        },
        "argmax_s2": {
          "n_clips": 50,
          "accuracy": 0.76,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 50,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.76,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              11,
              1,
              38,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 17.586281364234193,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 194
        },
        "frozen_B3_s0": {
          "n_clips": 50,
          "accuracy": 0.8,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 50,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.8,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              5,
              40,
              5,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 17.586281364234193,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 194
        },
        "frozen_B3_s1": {
          "n_clips": 50,
          "accuracy": 0.64,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 50,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.64,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              1,
              32,
              13,
              4,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 17.586281364234193,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 194
        },
        "frozen_B3_s2": {
          "n_clips": 50,
          "accuracy": 0.76,
          "class_support": {
            "PLAX": 0,
            "PSAX": 0,
            "A4C": 50,
            "A5C": 0,
            "SC4C": 0
          },
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.76,
            "A5C": null,
            "SC4C": null
          },
          "five_class_macro_f1": null,
          "confusion": [
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              11,
              1,
              38,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 17.586281364234193,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 194
        }
      }
    },
    "recounted_STFM_metrics": {
      "legacy_per_key_prior": {
        "original_nine_code": {
          "n_clips": 888,
          "accuracy": 0.9448198198198198,
          "macro_f1": 0.9093818329259263,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              57,
              0,
              0,
              0,
              0,
              15,
              0,
              0
            ],
            [
              0,
              0,
              44,
              0,
              0,
              0,
              1,
              5,
              0
            ],
            [
              0,
              0,
              0,
              30,
              0,
              0,
              0,
              11,
              0
            ],
            [
              0,
              0,
              0,
              0,
              225,
              6,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              2,
              38,
              0,
              0,
              0
            ],
            [
              0,
              3,
              1,
              0,
              0,
              0,
              84,
              0,
              0
            ],
            [
              0,
              0,
              0,
              4,
              0,
              0,
              0,
              52,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              1,
              0,
              0,
              34
            ]
          ]
        },
        "shared_five_family": {
          "n_clips": 800,
          "accuracy": 0.97,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 0.9315068493150684,
            "A4C": 0.974025974025974,
            "A5C": 0.95,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9653393303239074,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              204,
              0,
              0,
              0,
              15
            ],
            [
              0,
              0,
              225,
              6,
              0,
              0
            ],
            [
              0,
              0,
              2,
              38,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 800
        },
        "cut_note": "Whole-file classification emits no internal cut by construction; not a stability finding."
      },
      "corrected_prior_once_same_old_weights": {
        "original_nine_code": {
          "n_clips": 888,
          "accuracy": 0.9448198198198198,
          "macro_f1": 0.9093818329259263,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              57,
              0,
              0,
              0,
              0,
              15,
              0,
              0
            ],
            [
              0,
              0,
              44,
              0,
              0,
              0,
              1,
              5,
              0
            ],
            [
              0,
              0,
              0,
              30,
              0,
              0,
              0,
              11,
              0
            ],
            [
              0,
              0,
              0,
              0,
              225,
              6,
              0,
              0,
              0
            ],
            [
              0,
              0,
              0,
              0,
              2,
              38,
              0,
              0,
              0
            ],
            [
              0,
              3,
              1,
              0,
              0,
              0,
              84,
              0,
              0
            ],
            [
              0,
              0,
              0,
              4,
              0,
              0,
              0,
              52,
              0
            ],
            [
              0,
              0,
              0,
              0,
              0,
              1,
              0,
              0,
              34
            ]
          ]
        },
        "shared_five_family": {
          "n_clips": 800,
          "accuracy": 0.96875,
          "class_support": {
            "PLAX": 275,
            "PSAX": 219,
            "A4C": 231,
            "A5C": 40,
            "SC4C": 35
          },
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 0.9269406392694064,
            "A4C": 0.974025974025974,
            "A5C": 0.95,
            "SC4C": 0.9714285714285714
          },
          "five_class_macro_f1": 0.9648485905168421,
          "confusion": [
            [
              275,
              0,
              0,
              0,
              0,
              0
            ],
            [
              0,
              203,
              0,
              0,
              0,
              16
            ],
            [
              0,
              0,
              225,
              6,
              0,
              0
            ],
            [
              0,
              0,
              2,
              38,
              0,
              0
            ],
            [
              0,
              0,
              0,
              1,
              34,
              0
            ]
          ],
          "confusion_columns": [
            "PLAX",
            "PSAX",
            "A4C",
            "A5C",
            "SC4C",
            "unsupported"
          ],
          "internal_cuts": 0,
          "internal_cuts_per_clip": 0.0,
          "duration_s": 4624.666666666667,
          "internal_cuts_per_minute": 0.0,
          "n_samples": 800
        },
        "cut_note": "Whole-file classification emits no internal cut by construction; not a stability finding."
      }
    },
    "limits": [
      "EV9V is the historically accessed STFM source test, not external or held-out confirmation.",
      "Dynamic A4C and LVH PLAX are source-code recall checks, not EF or wall-thickness endpoints.",
      "CAMUS official 4CH source codes are not expert adjudication of A4C versus A5C.",
      "CAMUS half-cycle clips span approximately 0.18 to 0.51 seconds; short input support limits interpretation.",
      "EV9V clusters are sessions, not confirmed patients; Dynamic/LVH patient identities are unknown.",
      "CAMUS clusters are official patient identities; unknown pretraining/source overlap remains unknown.",
      "Clip labels are weak supervision; no clinical natural-transition or local-anatomy result is established.",
      "No source pixel reload or model forward is performed. Cached input-pixel digests are checked for form only.",
      "Media size/mtime, source-metadata hashes, video container timing and CAMUS header length are rechecked; these are not raw-media byte hashes.",
      "The prospective LVH timing correction retains v1 EV9V/Dynamic caches; LVH/CAMUS use the explicitly bound v2 protocol. Cache chronology uses filesystem timestamps, not recorded model-forward start times.",
      "Protocol pins only its declared source modules; unpinned transitive/library/environment dependencies cannot be retroactively frozen.",
      "Baseline caches have no per-case preprocessing/sampling/timing receipt; only logits shape, identity, manifest duration and declared input recipe can be audited.",
      "Per-case GPU timing is not stored in prediction caches; inference resource accounting requires the separate terminal/resource receipts.",
      "STFM whole-file classification emits no internal cuts by construction; zero cuts is not a stability result.",
      "Bootstrap intervals describe this fixed evaluation and grouping; they are not a clinical or finite-sample guarantee."
    ],
    "elapsed_wall_seconds": 13.332795870956033,
    "execution": "CPU only; no production decoder/metric imports; no model forward; aggregate-only artifacts"
  },
  "uncertainty": {
    "method": {
      "bootstrap_draws": 2000,
      "bootstrap_rng_seed": 20261030,
      "interval": "percentile 2.5%,97.5%; whole-group sampling with replacement",
      "case_weighting": "recordings within sampled groups retain their multiplicity",
      "seed_mean": "same groups/cases resampled for all three seeds; arithmetic mean of their metrics; no 3n pooling"
    },
    "datasets": {
      "ev9v": {
        "groups": 125,
        "n_cases": 800,
        "models": {
          "argmax_s0": {
            "correct": 784,
            "accuracy": 0.98,
            "accuracy_ci95": [
              0.9689181681681681,
              0.9888205684730657
            ],
            "macro_f1": 0.9537862999628717,
            "macro_f1_ci95": [
              0.9265025014712707,
              0.9735649986779907
            ],
            "class_recall": {
              "PLAX": 1.0,
              "PSAX": 1.0,
              "A4C": 0.9437229437229437,
              "A5C": 0.95,
              "SC4C": 0.9714285714285714
            },
            "cuts_per_clip": 0.11,
            "cuts_per_clip_ci95": [
              0.05917681673347723,
              0.17023750517498987
            ]
          },
          "argmax_s1": {
            "correct": 787,
            "accuracy": 0.98375,
            "accuracy_ci95": [
              0.9738899604755066,
              0.9918927103415663
            ],
            "macro_f1": 0.9585063893140198,
            "macro_f1_ci95": [
              0.9327435380728426,
              0.9788632361806713
            ],
            "class_recall": {
              "PLAX": 1.0,
              "PSAX": 1.0,
              "A4C": 0.974025974025974,
              "A5C": 0.85,
              "SC4C": 0.9714285714285714
            },
            "cuts_per_clip": 0.0925,
            "cuts_per_clip_ci95": [
              0.04760895883777239,
              0.14423543689320384
            ]
          },
          "argmax_s2": {
            "correct": 786,
            "accuracy": 0.9825,
            "accuracy_ci95": [
              0.9719620207896242,
              0.9907897762639187
            ],
            "macro_f1": 0.9585370541531347,
            "macro_f1_ci95": [
              0.9317336693702476,
              0.9778512360780716
            ],
            "class_recall": {
              "PLAX": 1.0,
              "PSAX": 1.0,
              "A4C": 0.9523809523809523,
              "A5C": 0.95,
              "SC4C": 0.9714285714285714
            },
            "cuts_per_clip": 0.12125,
            "cuts_per_clip_ci95": [
              0.06451036866359447,
              0.187602911727726
            ]
          },
          "frozen_B3_s0": {
            "correct": 782,
            "accuracy": 0.9775,
            "accuracy_ci95": [
              0.9654730574670416,
              0.9875
            ],
            "macro_f1": 0.9484549375583384,
            "macro_f1_ci95": [
              0.9204902689610867,
              0.9691976367302841
            ],
            "class_recall": {
              "PLAX": 1.0,
              "PSAX": 1.0,
              "A4C": 0.9393939393939394,
              "A5C": 0.925,
              "SC4C": 0.9714285714285714
            },
            "cuts_per_clip": 0.03125,
            "cuts_per_clip_ci95": [
              0.012345679012345678,
              0.05428282699174781
            ]
          },
          "frozen_B3_s1": {
            "correct": 785,
            "accuracy": 0.98125,
            "accuracy_ci95": [
              0.970974333738186,
              0.9898250795964556
            ],
            "macro_f1": 0.9541104574519246,
            "macro_f1_ci95": [
              0.9272087958004457,
              0.9742288802006409
            ],
            "class_recall": {
              "PLAX": 1.0,
              "PSAX": 1.0,
              "A4C": 0.961038961038961,
              "A5C": 0.875,
              "SC4C": 0.9714285714285714
            },
            "cuts_per_clip": 0.025,
            "cuts_per_clip_ci95": [
              0.012149565017843537,
              0.039370881119479624
            ]
          },
          "frozen_B3_s2": {
            "correct": 786,
            "accuracy": 0.9825,
            "accuracy_ci95": [
              0.9719620207896242,
              0.9907897762639187
            ],
            "macro_f1": 0.9585370541531347,
            "macro_f1_ci95": [
              0.9317336693702476,
              0.9778512360780716
            ],
            "class_recall": {
              "PLAX": 1.0,
              "PSAX": 1.0,
              "A4C": 0.9523809523809523,
              "A5C": 0.95,
              "SC4C": 0.9714285714285714
            },
            "cuts_per_clip": 0.02875,
            "cuts_per_clip_ci95": [
              0.013397075257500326,
              0.04551976652730056
            ]
          }
        },
        "mean_B3": {
          "accuracy": 0.9804166666666667,
          "ci95": [
            0.9696960119526491,
            0.9890910506938545
          ],
          "macro_f1": 0.9537008163877992,
          "macro_ci95": [
            0.927289354776908,
            0.9729133148774649
          ],
          "class_recall": {
            "PLAX": 1.0,
            "PSAX": 1.0,
            "A4C": 0.9509379509379509,
            "A5C": 0.9166666666666666,
            "SC4C": 0.9714285714285714
          },
          "class_recall_ci95": {
            "PLAX": [
              1.0,
              1.0
            ],
            "PSAX": [
              1.0,
              1.0
            ],
            "A4C": [
              0.9195388416902659,
              0.9758288880713695
            ],
            "A5C": [
              0.8455284552845529,
              0.9785114247311827
            ],
            "SC4C": [
              0.9,
              1.0
            ]
          },
          "cuts_per_clip": 0.028333333333333335,
          "cuts_per_clip_ci95": [
            0.013001844328023093,
            0.045360214423115566
          ]
        },
        "grouping": "session; patient identity unknown"
      },
      "dynamic": {
        "groups": 1277,
        "n_cases": 1277,
        "models": {
          "argmax_s0": {
            "correct": 1210,
            "accuracy": 0.9475332811276429,
            "accuracy_ci95": [
              0.9350039154267815,
              0.9592795614722005
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.9475332811276429,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.09945184025058731,
            "cuts_per_clip_ci95": [
              0.07515661707126077,
              0.12764291307752546
            ]
          },
          "argmax_s1": {
            "correct": 1209,
            "accuracy": 0.9467501957713391,
            "accuracy_ci95": [
              0.9342012529365702,
              0.9584964761158966
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.9467501957713391,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.11511354737666406,
            "cuts_per_clip_ci95": [
              0.0908379013312451,
              0.14252153484729835
            ]
          },
          "argmax_s2": {
            "correct": 1207,
            "accuracy": 0.9451840250587314,
            "accuracy_ci95": [
              0.93265465935787,
              0.9569303054032889
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.9451840250587314,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.09866875489428348,
            "cuts_per_clip_ci95": [
              0.07439310884886452,
              0.12685982772122162
            ]
          },
          "frozen_B3_s0": {
            "correct": 1211,
            "accuracy": 0.9483163664839468,
            "accuracy_ci95": [
              0.9357870007830854,
              0.9592795614722005
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.9483163664839468,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.007830853563038372,
            "cuts_per_clip_ci95": [
              0.0031323414252153485,
              0.01331245105716523
            ]
          },
          "frozen_B3_s1": {
            "correct": 1210,
            "accuracy": 0.9475332811276429,
            "accuracy_ci95": [
              0.934984338292874,
              0.9592795614722005
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.9475332811276429,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0031323414252153485,
            "cuts_per_clip_ci95": [
              0.0007830853563038371,
              0.006264682850430697
            ]
          },
          "frozen_B3_s2": {
            "correct": 1210,
            "accuracy": 0.9475332811276429,
            "accuracy_ci95": [
              0.9350039154267815,
              0.9592795614722005
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.9475332811276429,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.010180109631949883,
            "cuts_per_clip_ci95": [
              0.004698512137823023,
              0.015681284259984232
            ]
          }
        },
        "mean_B3": {
          "accuracy": 0.9477943095797442,
          "ci95": [
            0.936309057687288,
            0.9582354476637954
          ],
          "macro_f1": null,
          "macro_ci95": null,
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.9477943095797442,
            "A5C": null,
            "SC4C": null
          },
          "class_recall_ci95": {
            "PLAX": null,
            "PSAX": null,
            "A4C": [
              0.936309057687288,
              0.9582354476637954
            ],
            "A5C": null,
            "SC4C": null
          },
          "cuts_per_clip": 0.007047768206734535,
          "cuts_per_clip_ci95": [
            0.0033933698773166276,
            0.011224223440355
          ]
        },
        "grouping": "recording; patient identity unknown"
      },
      "lvh": {
        "groups": 343,
        "n_cases": 343,
        "models": {
          "argmax_s0": {
            "correct": 315,
            "accuracy": 0.9183673469387755,
            "accuracy_ci95": [
              0.8892128279883382,
              0.9446064139941691
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": 0.9183673469387755,
              "PSAX": null,
              "A4C": null,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.11078717201166181,
            "cuts_per_clip_ci95": [
              0.06705539358600583,
              0.16326530612244897
            ]
          },
          "argmax_s1": {
            "correct": 333,
            "accuracy": 0.9708454810495627,
            "accuracy_ci95": [
              0.9533527696793003,
              0.9883381924198251
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": 0.9708454810495627,
              "PSAX": null,
              "A4C": null,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.11661807580174927,
            "cuts_per_clip_ci95": [
              0.06997084548104957,
              0.17201166180758018
            ]
          },
          "argmax_s2": {
            "correct": 335,
            "accuracy": 0.9766763848396501,
            "accuracy_ci95": [
              0.9591836734693877,
              0.9912536443148688
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": 0.9766763848396501,
              "PSAX": null,
              "A4C": null,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.09620991253644315,
            "cuts_per_clip_ci95": [
              0.04956268221574344,
              0.1457725947521866
            ]
          },
          "frozen_B3_s0": {
            "correct": 317,
            "accuracy": 0.924198250728863,
            "accuracy_ci95": [
              0.8950437317784257,
              0.9504373177842566
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": 0.924198250728863,
              "PSAX": null,
              "A4C": null,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "frozen_B3_s1": {
            "correct": 335,
            "accuracy": 0.9766763848396501,
            "accuracy_ci95": [
              0.9591836734693877,
              0.9912536443148688
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": 0.9766763848396501,
              "PSAX": null,
              "A4C": null,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "frozen_B3_s2": {
            "correct": 333,
            "accuracy": 0.9708454810495627,
            "accuracy_ci95": [
              0.9532798833819242,
              0.9883381924198251
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": 0.9708454810495627,
              "PSAX": null,
              "A4C": null,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          }
        },
        "mean_B3": {
          "accuracy": 0.9572400388726919,
          "ci95": [
            0.9397473275024296,
            0.9727891156462585
          ],
          "macro_f1": null,
          "macro_ci95": null,
          "class_recall": {
            "PLAX": 0.9572400388726919,
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "class_recall_ci95": {
            "PLAX": [
              0.9397473275024296,
              0.9727891156462585
            ],
            "PSAX": null,
            "A4C": null,
            "A5C": null,
            "SC4C": null
          },
          "cuts_per_clip": 0.0,
          "cuts_per_clip_ci95": [
            0.0,
            0.0
          ]
        },
        "grouping": "recording; patient identity unknown"
      },
      "camus": {
        "groups": 50,
        "n_cases": 50,
        "models": {
          "argmax_s0": {
            "correct": 40,
            "accuracy": 0.8,
            "accuracy_ci95": [
              0.68,
              0.9
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.8,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "argmax_s1": {
            "correct": 32,
            "accuracy": 0.64,
            "accuracy_ci95": [
              0.5,
              0.76
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.64,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "argmax_s2": {
            "correct": 38,
            "accuracy": 0.76,
            "accuracy_ci95": [
              0.64,
              0.86
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.76,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "frozen_B3_s0": {
            "correct": 40,
            "accuracy": 0.8,
            "accuracy_ci95": [
              0.68,
              0.9
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.8,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "frozen_B3_s1": {
            "correct": 32,
            "accuracy": 0.64,
            "accuracy_ci95": [
              0.5,
              0.76
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.64,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          },
          "frozen_B3_s2": {
            "correct": 38,
            "accuracy": 0.76,
            "accuracy_ci95": [
              0.64,
              0.86
            ],
            "macro_f1": null,
            "macro_f1_ci95": null,
            "class_recall": {
              "PLAX": null,
              "PSAX": null,
              "A4C": 0.76,
              "A5C": null,
              "SC4C": null
            },
            "cuts_per_clip": 0.0,
            "cuts_per_clip_ci95": [
              0.0,
              0.0
            ]
          }
        },
        "mean_B3": {
          "accuracy": 0.7333333333333334,
          "ci95": [
            0.62,
            0.84
          ],
          "macro_f1": null,
          "macro_ci95": null,
          "class_recall": {
            "PLAX": null,
            "PSAX": null,
            "A4C": 0.7333333333333334,
            "A5C": null,
            "SC4C": null
          },
          "class_recall_ci95": {
            "PLAX": null,
            "PSAX": null,
            "A4C": [
              0.62,
              0.84
            ],
            "A5C": null,
            "SC4C": null
          },
          "cuts_per_clip": 0.0,
          "cuts_per_clip_ci95": [
            0.0,
            0.0
          ]
        },
        "grouping": "patient"
      }
    },
    "baseline": {
      "legacy_per_key_prior": {
        "correct": 776,
        "accuracy": 0.97,
        "accuracy_ci95": [
          0.9560849957365962,
          0.9823866067608044
        ],
        "macro_f1": 0.9653393303239074,
        "macro_f1_ci95": [
          0.9443289195031562,
          0.9817716679757521
        ],
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 0.9315068493150684,
          "A4C": 0.974025974025974,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "cuts_per_clip": 0.0,
        "cuts_per_clip_ci95": [
          0.0,
          0.0
        ],
        "groups": 125,
        "grouping": "session",
        "n_cases": 800,
        "original_nine_code": {
          "correct": 839,
          "accuracy": 0.9448198198198198,
          "accuracy_ci95": [
            0.9267702462854759,
            0.960972548465208
          ],
          "macro_f1": 0.9093818329259263,
          "macro_f1_ci95": [
            0.8804780393935637,
            0.9336983786706657
          ],
          "class_recall": {
            "PLHLA": 1.0,
            "PMASA": 0.7916666666666666,
            "PMVLSA": 0.88,
            "PASA": 0.7317073170731707,
            "A4C": 0.974025974025974,
            "A5C": 0.95,
            "PMPALA": 0.9545454545454546,
            "PPMLSA": 0.9285714285714286,
            "SC4C": 0.9714285714285714
          },
          "cuts_per_clip": 0.0,
          "cuts_per_clip_ci95": [
            0.0,
            0.0
          ],
          "groups": 127,
          "n_cases": 888,
          "grouping": "session"
        }
      },
      "corrected_prior_once_same_old_weights": {
        "correct": 775,
        "accuracy": 0.96875,
        "accuracy_ci95": [
          0.9551085861505177,
          0.981391262550737
        ],
        "macro_f1": 0.9648485905168421,
        "macro_f1_ci95": [
          0.9438606220276203,
          0.9810514346625023
        ],
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 0.9269406392694064,
          "A4C": 0.974025974025974,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "cuts_per_clip": 0.0,
        "cuts_per_clip_ci95": [
          0.0,
          0.0
        ],
        "groups": 125,
        "grouping": "session",
        "n_cases": 800,
        "original_nine_code": {
          "correct": 839,
          "accuracy": 0.9448198198198198,
          "accuracy_ci95": [
            0.9267702462854759,
            0.960972548465208
          ],
          "macro_f1": 0.9093818329259263,
          "macro_f1_ci95": [
            0.8804780393935637,
            0.9336983786706657
          ],
          "class_recall": {
            "PLHLA": 1.0,
            "PMASA": 0.7916666666666666,
            "PMVLSA": 0.88,
            "PASA": 0.7317073170731707,
            "A4C": 0.974025974025974,
            "A5C": 0.95,
            "PMPALA": 0.9545454545454546,
            "PPMLSA": 0.9285714285714286,
            "SC4C": 0.9714285714285714
          },
          "cuts_per_clip": 0.0,
          "cuts_per_clip_ci95": [
            0.0,
            0.0
          ],
          "groups": 127,
          "n_cases": 888,
          "grouping": "session"
        }
      }
    },
    "paired_original": {
      "legacy_per_key_prior": {
        "frozen_B3_s0": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.007500000000000062,
          "accuracy_difference_ci95": [
            -0.0072391765648525674,
            0.02329045305123534
          ],
          "macro_f1_difference": -0.016884392765568967,
          "macro_f1_difference_ci95": [
            -0.03932570270354843,
            0.002690093764338994
          ]
        },
        "frozen_B3_s1": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.011249999999999982,
          "accuracy_difference_ci95": [
            -0.0022165460641090317,
            0.02575601018591124
          ],
          "macro_f1_difference": -0.01122887287198282,
          "macro_f1_difference_ci95": [
            -0.033380798361273875,
            0.007979311711727154
          ]
        },
        "frozen_B3_s2": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.012500000000000067,
          "accuracy_difference_ci95": [
            -0.001212951432129522,
            0.027058327125353412
          ],
          "macro_f1_difference": -0.00680227617077267,
          "macro_f1_difference_ci95": [
            -0.026171721212643335,
            0.010349726348247679
          ]
        },
        "argmax_s0": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.010000000000000009,
          "accuracy_difference_ci95": [
            -0.003937396906414359,
            0.025159111979753925
          ],
          "macro_f1_difference": -0.011553030361035721,
          "macro_f1_difference_ci95": [
            -0.032226579140136276,
            0.00699575924673786
          ]
        },
        "argmax_s1": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.01375000000000004,
          "accuracy_difference_ci95": [
            0.0,
            0.028185792047364457
          ],
          "macro_f1_difference": -0.0068329410098876275,
          "macro_f1_difference_ci95": [
            -0.029927153471960174,
            0.013005946049379066
          ]
        },
        "argmax_s2": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.012500000000000067,
          "accuracy_difference_ci95": [
            -0.001212951432129522,
            0.027058327125353412
          ],
          "macro_f1_difference": -0.00680227617077267,
          "macro_f1_difference_ci95": [
            -0.026171721212643335,
            0.010349726348247679
          ]
        },
        "mean_B3": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.01041666666666674,
          "accuracy_difference_ci95": [
            -0.003162784031686394,
            0.02504975957647753
          ],
          "macro_f1_difference": -0.011638513936108152,
          "macro_f1_difference_ci95": [
            -0.031128556970807696,
            0.006529581212555601
          ]
        }
      },
      "corrected_prior_once_same_old_weights": {
        "frozen_B3_s0": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.008750000000000036,
          "accuracy_difference_ci95": [
            -0.006267250404492217,
            0.024739905631188153
          ],
          "macro_f1_difference": -0.0163936529585037,
          "macro_f1_difference_ci95": [
            -0.0388257720054002,
            0.003157117813562867
          ]
        },
        "frozen_B3_s1": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.012499999999999956,
          "accuracy_difference_ci95": [
            -0.0013495732574680063,
            0.027311049122736884
          ],
          "macro_f1_difference": -0.010738133064917554,
          "macro_f1_difference_ci95": [
            -0.03310000570357144,
            0.008712765876509192
          ]
        },
        "frozen_B3_s2": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.01375000000000004,
          "accuracy_difference_ci95": [
            0.0,
            0.0285732076143035
          ],
          "macro_f1_difference": -0.006311536363707404,
          "macro_f1_difference_ci95": [
            -0.02561032580974802,
            0.010933829440454148
          ]
        },
        "argmax_s0": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.011249999999999982,
          "accuracy_difference_ci95": [
            -0.002874823553135736,
            0.026698835761483464
          ],
          "macro_f1_difference": -0.011062290553970455,
          "macro_f1_difference_ci95": [
            -0.031715187546153434,
            0.007432769407829613
          ]
        },
        "argmax_s1": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.015000000000000013,
          "accuracy_difference_ci95": [
            0.0012870012870012104,
            0.029851192068332148
          ],
          "macro_f1_difference": -0.006342201202822362,
          "macro_f1_difference_ci95": [
            -0.02939299924159245,
            0.013557039659754643
          ]
        },
        "argmax_s2": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.01375000000000004,
          "accuracy_difference_ci95": [
            0.0,
            0.0285732076143035
          ],
          "macro_f1_difference": -0.006311536363707404,
          "macro_f1_difference_ci95": [
            -0.02561032580974802,
            0.010933829440454148
          ]
        },
        "mean_B3": {
          "n_cases": 800,
          "groups": 125,
          "grouping": "paired sessions; identical case membership",
          "accuracy_difference": 0.011666666666666714,
          "accuracy_difference_ci95": [
            -0.0021355626070874213,
            0.026810520939734326
          ],
          "macro_f1_difference": -0.011147774129042887,
          "macro_f1_difference_ci95": [
            -0.03071250312010729,
            0.006890299335647024
          ]
        }
      }
    },
    "limits": [
      "EV9V is the historically accessed STFM source test, not external or held-out confirmation.",
      "Dynamic A4C and LVH PLAX are source-code recall checks, not EF or wall-thickness endpoints.",
      "CAMUS official 4CH source codes are not expert adjudication of A4C versus A5C.",
      "CAMUS half-cycle clips span approximately 0.18 to 0.51 seconds; short input support limits interpretation.",
      "EV9V clusters are sessions, not confirmed patients; Dynamic/LVH patient identities are unknown.",
      "CAMUS clusters are official patient identities; unknown pretraining/source overlap remains unknown.",
      "Clip labels are weak supervision; no clinical natural-transition or local-anatomy result is established.",
      "No source pixel reload or model forward is performed. Cached input-pixel digests are checked for form only.",
      "Media size/mtime, source-metadata hashes, video container timing and CAMUS header length are rechecked; these are not raw-media byte hashes.",
      "The prospective LVH timing correction retains v1 EV9V/Dynamic caches; LVH/CAMUS use the explicitly bound v2 protocol. Cache chronology uses filesystem timestamps, not recorded model-forward start times.",
      "Protocol pins only its declared source modules; unpinned transitive/library/environment dependencies cannot be retroactively frozen.",
      "Baseline caches have no per-case preprocessing/sampling/timing receipt; only logits shape, identity, manifest duration and declared input recipe can be audited.",
      "Per-case GPU timing is not stored in prediction caches; inference resource accounting requires the separate terminal/resource receipts.",
      "STFM whole-file classification emits no internal cuts by construction; zero cuts is not a stability result.",
      "Bootstrap intervals describe this fixed evaluation and grouping; they are not a clinical or finite-sample guarantee."
    ]
  },
  "native_results": {
    "ev9v": {
      "argmax_s0": {
        "n_clips": 800,
        "accuracy": 0.98,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 1.0,
          "A4C": 0.9437229437229437,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9537862999628717,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            219,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            218,
            11,
            2,
            0
          ],
          [
            0,
            0,
            2,
            38,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 88,
        "internal_cuts_per_clip": 0.11,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 1.141703906587862,
        "n_samples": 46507
      },
      "argmax_s1": {
        "n_clips": 800,
        "accuracy": 0.98375,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 1.0,
          "A4C": 0.974025974025974,
          "A5C": 0.85,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9585063893140198,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            219,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            225,
            4,
            2,
            0
          ],
          [
            0,
            0,
            6,
            34,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 74,
        "internal_cuts_per_clip": 0.0925,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 0.9600691941761568,
        "n_samples": 46507
      },
      "argmax_s2": {
        "n_clips": 800,
        "accuracy": 0.9825,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 1.0,
          "A4C": 0.9523809523809523,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9585370541531347,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            219,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            220,
            9,
            2,
            0
          ],
          [
            0,
            0,
            2,
            38,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 97,
        "internal_cuts_per_clip": 0.12125,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 1.25846907885253,
        "n_samples": 46507
      },
      "frozen_B3_s0": {
        "n_clips": 800,
        "accuracy": 0.9775,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 1.0,
          "A4C": 0.9393939393939394,
          "A5C": 0.925,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9484549375583384,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            219,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            217,
            12,
            2,
            0
          ],
          [
            0,
            0,
            3,
            37,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 25,
        "internal_cuts_per_clip": 0.03125,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 0.3243477007351881,
        "n_samples": 46507
      },
      "frozen_B3_s1": {
        "n_clips": 800,
        "accuracy": 0.98125,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 1.0,
          "A4C": 0.961038961038961,
          "A5C": 0.875,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9541104574519246,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            219,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            222,
            7,
            2,
            0
          ],
          [
            0,
            0,
            5,
            35,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 20,
        "internal_cuts_per_clip": 0.025,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 0.2594781605881505,
        "n_samples": 46507
      },
      "frozen_B3_s2": {
        "n_clips": 800,
        "accuracy": 0.9825,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 1.0,
          "A4C": 0.9523809523809523,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9585370541531347,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            219,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            220,
            9,
            2,
            0
          ],
          [
            0,
            0,
            2,
            38,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 23,
        "internal_cuts_per_clip": 0.02875,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 0.2983998846763731,
        "n_samples": 46507
      }
    },
    "dynamic": {
      "argmax_s0": {
        "n_clips": 1277,
        "accuracy": 0.9475332811276429,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 1277,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.9475332811276429,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            1,
            1210,
            64,
            2,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 127,
        "internal_cuts_per_clip": 0.09945184025058731,
        "duration_s": 4403.130137983133,
        "internal_cuts_per_minute": 1.7305870508497767,
        "n_samples": 44537
      },
      "argmax_s1": {
        "n_clips": 1277,
        "accuracy": 0.9467501957713391,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 1277,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.9467501957713391,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            2,
            1209,
            63,
            3,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 147,
        "internal_cuts_per_clip": 0.11511354737666406,
        "duration_s": 4403.130137983133,
        "internal_cuts_per_minute": 2.003120444684387,
        "n_samples": 44537
      },
      "argmax_s2": {
        "n_clips": 1277,
        "accuracy": 0.9451840250587314,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 1277,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.9451840250587314,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            1,
            1207,
            61,
            8,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 126,
        "internal_cuts_per_clip": 0.09866875489428348,
        "duration_s": 4403.130137983133,
        "internal_cuts_per_minute": 1.716960381158046,
        "n_samples": 44537
      },
      "frozen_B3_s0": {
        "n_clips": 1277,
        "accuracy": 0.9483163664839468,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 1277,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.9483163664839468,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            1,
            1211,
            62,
            3,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 10,
        "internal_cuts_per_clip": 0.007830853563038372,
        "duration_s": 4403.130137983133,
        "internal_cuts_per_minute": 0.13626669691730525,
        "n_samples": 44537
      },
      "frozen_B3_s1": {
        "n_clips": 1277,
        "accuracy": 0.9475332811276429,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 1277,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.9475332811276429,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            2,
            1210,
            63,
            2,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 4,
        "internal_cuts_per_clip": 0.0031323414252153485,
        "duration_s": 4403.130137983133,
        "internal_cuts_per_minute": 0.0545066787669221,
        "n_samples": 44537
      },
      "frozen_B3_s2": {
        "n_clips": 1277,
        "accuracy": 0.9475332811276429,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 1277,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.9475332811276429,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            1210,
            59,
            8,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 13,
        "internal_cuts_per_clip": 0.010180109631949883,
        "duration_s": 4403.130137983133,
        "internal_cuts_per_minute": 0.17714670599249682,
        "n_samples": 44537
      }
    },
    "lvh": {
      "argmax_s0": {
        "n_clips": 343,
        "accuracy": 0.9183673469387755,
        "class_support": {
          "PLAX": 343,
          "PSAX": 0,
          "A4C": 0,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": 0.9183673469387755,
          "PSAX": null,
          "A4C": null,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            315,
            4,
            9,
            14,
            1,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 38,
        "internal_cuts_per_clip": 0.11078717201166181,
        "duration_s": 1174.7972042670065,
        "internal_cuts_per_minute": 1.9407604918693733,
        "n_samples": 11884
      },
      "argmax_s1": {
        "n_clips": 343,
        "accuracy": 0.9708454810495627,
        "class_support": {
          "PLAX": 343,
          "PSAX": 0,
          "A4C": 0,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": 0.9708454810495627,
          "PSAX": null,
          "A4C": null,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            333,
            2,
            5,
            2,
            1,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 40,
        "internal_cuts_per_clip": 0.11661807580174927,
        "duration_s": 1174.7972042670065,
        "internal_cuts_per_minute": 2.0429057809151296,
        "n_samples": 11884
      },
      "argmax_s2": {
        "n_clips": 343,
        "accuracy": 0.9766763848396501,
        "class_support": {
          "PLAX": 343,
          "PSAX": 0,
          "A4C": 0,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": 0.9766763848396501,
          "PSAX": null,
          "A4C": null,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            335,
            2,
            5,
            1,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 33,
        "internal_cuts_per_clip": 0.09620991253644315,
        "duration_s": 1174.7972042670065,
        "internal_cuts_per_minute": 1.6853972692549821,
        "n_samples": 11884
      },
      "frozen_B3_s0": {
        "n_clips": 343,
        "accuracy": 0.924198250728863,
        "class_support": {
          "PLAX": 343,
          "PSAX": 0,
          "A4C": 0,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": 0.924198250728863,
          "PSAX": null,
          "A4C": null,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            317,
            3,
            9,
            13,
            1,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 1174.7972042670065,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 11884
      },
      "frozen_B3_s1": {
        "n_clips": 343,
        "accuracy": 0.9766763848396501,
        "class_support": {
          "PLAX": 343,
          "PSAX": 0,
          "A4C": 0,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": 0.9766763848396501,
          "PSAX": null,
          "A4C": null,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            335,
            2,
            4,
            1,
            1,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 1174.7972042670065,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 11884
      },
      "frozen_B3_s2": {
        "n_clips": 343,
        "accuracy": 0.9708454810495627,
        "class_support": {
          "PLAX": 343,
          "PSAX": 0,
          "A4C": 0,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": 0.9708454810495627,
          "PSAX": null,
          "A4C": null,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            333,
            1,
            8,
            1,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 1174.7972042670065,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 11884
      }
    },
    "camus": {
      "argmax_s0": {
        "n_clips": 50,
        "accuracy": 0.8,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 50,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.8,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            5,
            40,
            5,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 17.586281364234193,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 194
      },
      "argmax_s1": {
        "n_clips": 50,
        "accuracy": 0.64,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 50,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.64,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            1,
            32,
            13,
            4,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 17.586281364234193,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 194
      },
      "argmax_s2": {
        "n_clips": 50,
        "accuracy": 0.76,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 50,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.76,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            11,
            1,
            38,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 17.586281364234193,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 194
      },
      "frozen_B3_s0": {
        "n_clips": 50,
        "accuracy": 0.8,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 50,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.8,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            5,
            40,
            5,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 17.586281364234193,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 194
      },
      "frozen_B3_s1": {
        "n_clips": 50,
        "accuracy": 0.64,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 50,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.64,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            1,
            32,
            13,
            4,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 17.586281364234193,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 194
      },
      "frozen_B3_s2": {
        "n_clips": 50,
        "accuracy": 0.76,
        "class_support": {
          "PLAX": 0,
          "PSAX": 0,
          "A4C": 50,
          "A5C": 0,
          "SC4C": 0
        },
        "class_recall": {
          "PLAX": null,
          "PSAX": null,
          "A4C": 0.76,
          "A5C": null,
          "SC4C": null
        },
        "five_class_macro_f1": null,
        "confusion": [
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            11,
            1,
            38,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 17.586281364234193,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 194
      }
    }
  },
  "stfm_original_results": {
    "legacy_per_key_prior": {
      "original_nine_code": {
        "n_clips": 888,
        "accuracy": 0.9448198198198198,
        "macro_f1": 0.9093818329259263,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            57,
            0,
            0,
            0,
            0,
            15,
            0,
            0
          ],
          [
            0,
            0,
            44,
            0,
            0,
            0,
            1,
            5,
            0
          ],
          [
            0,
            0,
            0,
            30,
            0,
            0,
            0,
            11,
            0
          ],
          [
            0,
            0,
            0,
            0,
            225,
            6,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            2,
            38,
            0,
            0,
            0
          ],
          [
            0,
            3,
            1,
            0,
            0,
            0,
            84,
            0,
            0
          ],
          [
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            52,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            34
          ]
        ]
      },
      "shared_five_family": {
        "n_clips": 800,
        "accuracy": 0.97,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 0.9315068493150684,
          "A4C": 0.974025974025974,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9653393303239074,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            204,
            0,
            0,
            0,
            15
          ],
          [
            0,
            0,
            225,
            6,
            0,
            0
          ],
          [
            0,
            0,
            2,
            38,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 800
      },
      "cut_note": "Whole-file classification emits no internal cut by construction; not a stability finding."
    },
    "corrected_prior_once_same_old_weights": {
      "original_nine_code": {
        "n_clips": 888,
        "accuracy": 0.9448198198198198,
        "macro_f1": 0.9093818329259263,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            57,
            0,
            0,
            0,
            0,
            15,
            0,
            0
          ],
          [
            0,
            0,
            44,
            0,
            0,
            0,
            1,
            5,
            0
          ],
          [
            0,
            0,
            0,
            30,
            0,
            0,
            0,
            11,
            0
          ],
          [
            0,
            0,
            0,
            0,
            225,
            6,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0,
            2,
            38,
            0,
            0,
            0
          ],
          [
            0,
            3,
            1,
            0,
            0,
            0,
            84,
            0,
            0
          ],
          [
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            52,
            0
          ],
          [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            34
          ]
        ]
      },
      "shared_five_family": {
        "n_clips": 800,
        "accuracy": 0.96875,
        "class_support": {
          "PLAX": 275,
          "PSAX": 219,
          "A4C": 231,
          "A5C": 40,
          "SC4C": 35
        },
        "class_recall": {
          "PLAX": 1.0,
          "PSAX": 0.9269406392694064,
          "A4C": 0.974025974025974,
          "A5C": 0.95,
          "SC4C": 0.9714285714285714
        },
        "five_class_macro_f1": 0.9648485905168421,
        "confusion": [
          [
            275,
            0,
            0,
            0,
            0,
            0
          ],
          [
            0,
            203,
            0,
            0,
            0,
            16
          ],
          [
            0,
            0,
            225,
            6,
            0,
            0
          ],
          [
            0,
            0,
            2,
            38,
            0,
            0
          ],
          [
            0,
            0,
            0,
            1,
            34,
            0
          ]
        ],
        "confusion_columns": [
          "PLAX",
          "PSAX",
          "A4C",
          "A5C",
          "SC4C",
          "unsupported"
        ],
        "internal_cuts": 0,
        "internal_cuts_per_clip": 0.0,
        "duration_s": 4624.666666666667,
        "internal_cuts_per_minute": 0.0,
        "n_samples": 800
      },
      "cut_note": "Whole-file classification emits no internal cut by construction; not a stability finding."
    }
  },
  "source_hashes": {
    "native_results.json": "fa5b7f7ad05d8716f783586c412a5b7696ecdc1a3ac513c7e062ea1d5b3c7ae9",
    "stfm_native_results.json": "4e9ac80851b6a9a6fbff432a984ffbb9b20c5326bb75bb7250370f8d0ffe8c21",
    "verification.json": "c09ce664e4ee857ef7a78c18f602b0a7b2c464090bb8daf8da65ee31eec7da7e",
    "uncertainty.json": "5af98b90fd3ecd64e90636c56ef028ae065b09b6f2acdb5d359f713124fe82bb",
    "availability.json": "b50aea5e5b5ffe712a9bc76024bcc61e0fd221f8dcc98413cb7d4902e7c0a0b4",
    "input_audit.json": "322133d4c220f5ea53ef82ddbd92e4325e6286e0a1bf0b2e9905e989e1dd4bc2",
    "resources.json": "646b69f3da09fb14b5e6773d4e4d2393924ab6e045183fc35da120a63252b439"
  }
};
