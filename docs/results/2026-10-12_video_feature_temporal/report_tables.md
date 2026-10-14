# Video feature temporal round: tables

Recompute: `python scripts/report_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml --evaluate`.
Each cell gives the selection half, then the calibration half.

## Systems

| System | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|
| Linear + B3 s0 | 0.947 / 0.977 | 0.955 / 0.975 | 0.953 / 0.975 | 0.9786 / 0.9861 | 0.046 / 0.013 | 0.026 / 0.020 | 2 / 2 | 0 / 0 |
| Linear argmax s0 | 0.901 / 0.941 | 0.974 / 0.980 | 0.834 / 0.909 | 0.9701 / 0.9861 | 0.134 / 0.118 | 0.039 / 0.040 | 3 / 2 | 0 / 0 |
| MS-TCN s0 | 0.920 / 0.967 | 0.966 / 0.962 | 0.918 / 0.962 | 0.9626 / 0.9780 | 0.074 / 0.038 | 0.035 / 0.025 | 0 / 1 | 3 / 2 |
| ASRF s0 | 0.937 / 0.980 | 0.948 / 0.970 | 0.976 / 0.992 | 0.9869 / 0.9780 | 0.014 / 0.008 | 0.035 / 0.015 | 0 / 1 | 1 / 2 |
| ASRF frame branch s0 | 0.919 / 0.969 | 0.953 / 0.962 | 0.919 / 0.967 | 0.9869 / 0.9780 | 0.063 / 0.038 | 0.030 / 0.025 | 0 / 1 | 1 / 2 |
| Linear + B3 s1 | 0.957 / 0.978 | 0.953 / 0.980 | 0.971 / 0.973 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.009 / 0.015 | 2 / 2 | 0 / 0 |
| Linear argmax s1 | 0.907 / 0.958 | 0.979 / 0.985 | 0.858 / 0.938 | 0.9701 / 0.9732 | 0.155 / 0.080 | 0.035 / 0.040 | 3 / 4 | 0 / 0 |
| MS-TCN s1 | 0.953 / 0.974 | 0.963 / 0.977 | 0.951 / 0.975 | 0.9771 / 0.9780 | 0.042 / 0.051 | 0.017 / 0.025 | 1 / 1 | 1 / 2 |
| ASRF s1 | 0.948 / 0.975 | 0.948 / 0.972 | 0.978 / 0.992 | 0.9771 / 0.9723 | 0.004 / 0.008 | 0.030 / 0.010 | 1 / 3 | 1 / 1 |
| ASRF frame branch s1 | 0.934 / 0.974 | 0.950 / 0.977 | 0.914 / 0.975 | 0.9869 / 0.9647 | 0.049 / 0.059 | 0.022 / 0.020 | 0 / 3 | 1 / 2 |
| Linear + B3 s2 | 0.939 / 0.973 | 0.971 / 0.972 | 0.912 / 0.953 | 0.9786 / 0.9861 | 0.085 / 0.042 | 0.030 / 0.030 | 2 / 2 | 0 / 0 |
| Linear argmax s2 | 0.908 / 0.950 | 0.982 / 0.975 | 0.841 / 0.917 | 0.9786 / 0.9796 | 0.162 / 0.131 | 0.035 / 0.040 | 2 / 3 | 0 / 0 |
| MS-TCN s2 | 0.943 / 0.963 | 0.958 / 0.970 | 0.938 / 0.953 | 0.9753 / 0.9780 | 0.056 / 0.055 | 0.013 / 0.035 | 0 / 1 | 2 / 2 |
| ASRF s2 | 0.951 / 0.970 | 0.945 / 0.957 | 0.978 / 0.984 | 0.9869 / 0.9929 | 0.000 / 0.008 | 0.030 / 0.015 | 0 / 1 | 1 / 0 |
| ASRF frame branch s2 | 0.937 / 0.966 | 0.958 / 0.962 | 0.929 / 0.965 | 0.9869 / 0.9929 | 0.074 / 0.051 | 0.026 / 0.020 | 0 / 1 | 1 / 0 |
| frozen Ordered + B3 s0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.9786 / 0.9929 | 0.042 / 0.025 | 0.017 / 0.025 | 2 / 1 | 0 / 0 |
| frozen Ordered argmax s0 | 0.893 / 0.961 | 0.979 / 0.987 | 0.846 / 0.942 | 0.9701 / 0.9861 | 0.095 / 0.093 | 0.043 / 0.035 | 3 / 2 | 0 / 0 |
| frozen Ordered + B3 s1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.004 / 0.025 | 2 / 2 | 0 / 0 |
| frozen Ordered argmax s1 | 0.925 / 0.961 | 0.984 / 0.990 | 0.872 / 0.945 | 0.9877 / 0.9796 | 0.113 / 0.101 | 0.030 / 0.040 | 1 / 3 | 0 / 0 |
| frozen Ordered + B3 s2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.9701 / 0.9861 | 0.046 / 0.013 | 0.004 / 0.025 | 3 / 2 | 0 / 0 |
| frozen Ordered argmax s2 | 0.926 / 0.960 | 0.984 / 0.982 | 0.874 / 0.940 | 0.9701 / 0.9861 | 0.116 / 0.097 | 0.026 / 0.035 | 3 / 2 | 0 / 0 |
| E0 + B3 | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.9650 / 0.9660 | 0.014 / 0.034 | 0.004 / 0.020 | 0 / 0 | 3 / 4 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.9780 / 0.9856 | 0.046 / 0.068 | 0.039 / 0.020 | 0 / 1 | 1 / 1 |
| B7 + B3 | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.9566 / 0.9927 | 0.063 / 0.042 | 0.057 / 0.035 | 1 / 0 | 2 / 1 |
| E1 + ASRF s0 | 0.910 / 0.976 | 0.942 / 0.977 | 0.952 / 0.980 | 0.9477 / 0.9929 | 0.046 / 0.038 | 0.030 / 0.015 | 2 / 1 | 2 / 0 |
| E1 + ASRF s1 | 0.909 / 0.964 | 0.950 / 0.972 | 0.936 / 0.970 | 0.9664 / 0.9753 | 0.007 / 0.030 | 0.057 / 0.030 | 0 / 1 | 2 / 2 |
| E1 + ASRF s2 | 0.907 / 0.964 | 0.953 / 0.977 | 0.933 / 0.970 | 0.9664 / 0.9753 | 0.021 / 0.046 | 0.057 / 0.035 | 0 / 1 | 2 / 2 |
| Original FACT s0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.9886 / 0.9856 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| Original FACT s1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.9780 / 0.9856 | 0.000 / 0.030 | 0.043 / 0.030 | 0 / 1 | 1 / 1 |
| Original FACT s2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.9664 / 0.9929 | 0.000 / 0.034 | 0.035 / 0.025 | 0 / 1 | 2 / 0 |
| paired ASRF s0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.9304 / 0.9856 | 0.011 / 0.004 | 0.065 / 0.035 | 1 / 1 | 4 / 1 |
| paired ASRF s1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.9537 / 0.9856 | 0.007 / 0.004 | 0.048 / 0.035 | 0 / 1 | 3 / 1 |
| paired ASRF s2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.9537 / 0.9856 | 0.011 / 0.004 | 0.065 / 0.030 | 0 / 1 | 3 / 1 |

## Comparisons

| Comparison | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict | Other consistent differences |
|---|---|---|---|---|
| ASRF - (Linear + B3) | selection | -0.0107 [-0.0273, +0.0055]; -0.0093 [-0.0340, +0.0128]; +0.0118 [-0.0067, +0.0303] | no consistent difference detected | none |
| ASRF - (Linear + B3) | calibration | +0.0027 [-0.0088, +0.0143]; -0.0032 [-0.0179, +0.0102]; -0.0034 [-0.0212, +0.0112] | no consistent difference detected | none |
| ASRF - MS-TCN | selection | +0.0166 [-0.0031, +0.0391]; -0.0058 [-0.0250, +0.0132]; +0.0079 [-0.0058, +0.0220] | no consistent difference detected | native cuts per clip consistent negative |
| ASRF - MS-TCN | calibration | +0.0124 [-0.0011, +0.0272]; +0.0007 [-0.0099, +0.0106]; +0.0071 [-0.0038, +0.0192] | no consistent difference detected | none |
| MS-TCN - (Linear + B3) | selection | -0.0273 [-0.0467, -0.0106]; -0.0035 [-0.0208, +0.0135]; +0.0039 [-0.0147, +0.0226] | no consistent difference detected | none |
| MS-TCN - (Linear + B3) | calibration | -0.0097 [-0.0242, +0.0028]; -0.0039 [-0.0185, +0.0093]; -0.0106 [-0.0277, +0.0047] | no consistent difference detected | none |
| ASRF - (frozen Ordered + B3) | selection | -0.0019 [-0.0221, +0.0181]; -0.0127 [-0.0400, +0.0104]; -0.0138 [-0.0328, +0.0037] | no consistent difference detected | native cuts per clip consistent negative |
| ASRF - (frozen Ordered + B3) | calibration | -0.0022 [-0.0132, +0.0100]; -0.0089 [-0.0225, +0.0037]; -0.0100 [-0.0255, +0.0028] | no consistent difference detected | none |
| MS-TCN - (frozen Ordered + B3) | selection | -0.0186 [-0.0417, +0.0037]; -0.0069 [-0.0242, +0.0092]; -0.0217 [-0.0420, -0.0032] | no consistent difference detected | none |
| MS-TCN - (frozen Ordered + B3) | calibration | -0.0145 [-0.0286, -0.0029]; -0.0097 [-0.0230, +0.0026]; -0.0171 [-0.0345, -0.0008] | no consistent difference detected | none |
| (Linear + B3) - (frozen Ordered + B3) | selection | +0.0088 [-0.0058, +0.0231]; -0.0034 [-0.0128, +0.0052]; -0.0256 [-0.0446, -0.0093] | no consistent difference detected | none |
| (Linear + B3) - (frozen Ordered + B3) | calibration | -0.0049 [-0.0121, +0.0011]; -0.0058 [-0.0118, -0.0001]; -0.0066 [-0.0144, +0.0000] | no consistent difference detected | none |
| ASRF - paired ASRF | selection | +0.0587 [+0.0285, +0.0901]; +0.0600 [+0.0271, +0.0960]; +0.0681 [+0.0366, +0.1003] | consistent positive | none |
| ASRF - paired ASRF | calibration | +0.0074 [-0.0064, +0.0217]; +0.0026 [-0.0105, +0.0153]; +0.0035 [-0.0144, +0.0199] | no consistent difference detected | none |
| ASRF - (E1 + B3) | selection | +0.0426 [+0.0109, +0.0747]; +0.0536 [+0.0165, +0.0898]; +0.0566 [+0.0246, +0.0891] | consistent positive | none |
| ASRF - (E1 + B3) | calibration | +0.0206 [+0.0034, +0.0389]; +0.0158 [-0.0021, +0.0329]; +0.0108 [-0.0094, +0.0282] | no consistent difference detected | native cuts per clip consistent negative |
| ASRF - Original FACT | selection | +0.0303 [+0.0028, +0.0585]; +0.0493 [+0.0168, +0.0827]; +0.0397 [+0.0089, +0.0723] | consistent positive | multi boundary F1 consistent positive |
| ASRF - Original FACT | calibration | +0.0122 [-0.0070, +0.0325]; +0.0142 [-0.0031, +0.0331]; +0.0045 [-0.0135, +0.0208] | no consistent difference detected | none |
| ASRF - historical E1 + ASRF | selection | +0.0266 [-0.0009, +0.0546]; +0.0387 [+0.0048, +0.0740]; +0.0434 [+0.0135, +0.0752] | no consistent difference detected | none |
| ASRF - historical E1 + ASRF | calibration | +0.0037 [-0.0097, +0.0179]; +0.0104 [-0.0031, +0.0229]; +0.0054 [-0.0136, +0.0250] | no consistent difference detected | none |

## Goal

| System | Three-seed mean | Margin to the minimum | Calibration sensitivity margin | Constraint failures | Goal met |
|---|---|---|---|---|---|
| Linear + B3 | 0.9476 / 0.9760 | +0.0032 / -0.0062 | -0.0079 | 0 | False |
| MS-TCN | 0.9387 / 0.9679 | -0.0057 / -0.0142 | -0.0160 | 4 | False |
| ASRF | 0.9449 / 0.9747 | +0.0005 / -0.0075 | -0.0092 | 5 | False |
| frozen Ordered + B3 | 0.9544 / 0.9817 | +0.0100 / -0.0004 | -0.0022 | 1 | False |

## Linear + B3 settings

```
{
 "0": {
  "status": "feasible",
  "params": {
   "half_window": 7,
   "threshold": 0.1730087995635474,
   "nms_radius": 5,
   "threshold_quantile": 0.999
  },
  "n_feasible": 23
 },
 "1": {
  "status": "feasible",
  "params": {
   "half_window": 7,
   "threshold": 0.19467906761053383,
   "nms_radius": 5,
   "threshold_quantile": 0.999
  },
  "n_feasible": 6
 },
 "2": {
  "status": "feasible",
  "params": {
   "half_window": 9,
   "threshold": 0.10074527008505041,
   "nms_radius": 5,
   "threshold_quantile": 0.995
  },
  "n_feasible": 3
 }
}
```

## Frozen diagnostic list

```
{
 "s0|selection": {
  "Linear + B3": {
   "listed": {
    "units": 16,
    "recordings": 16,
    "sessions": 12
   },
   "fixed": {
    "units": 3,
    "recordings": 4,
    "sessions": 4
   },
   "new_misses": {
    "units": 4,
    "recordings": 5,
    "sessions": 5
   }
  },
  "MS-TCN": {
   "listed": {
    "units": 16,
    "recordings": 16,
    "sessions": 12
   },
   "fixed": {
    "units": 6,
    "recordings": 7,
    "sessions": 7
   },
   "new_misses": {
    "units": 12,
    "recordings": 14,
    "sessions": 10
   }
  },
  "ASRF": {
   "listed": {
    "units": 16,
    "recordings": 16,
    "sessions": 12
   },
   "fixed": {
    "units": 4,
    "recordings": 5,
    "sessions": 5
   },
   "new_misses": {
    "units": 13,
    "recordings": 15,
    "sessions": 11
   }
  }
 },
 "s0|calibration": {
  "Linear + B3": {
   "listed": {
    "units": 3,
    "recordings": 5,
    "sessions": 4
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 5,
    "recordings": 5,
    "sessions": 5
   }
  },
  "MS-TCN": {
   "listed": {
    "units": 3,
    "recordings": 5,
    "sessions": 4
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 10,
    "recordings": 12,
    "sessions": 11
   }
  },
  "ASRF": {
   "listed": {
    "units": 3,
    "recordings": 5,
    "sessions": 4
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 9,
    "recordings": 11,
    "sessions": 8
   }
  }
 },
 "s1|selection": {
  "Linear + B3": {
   "listed": {
    "units": 11,
    "recordings": 13,
    "sessions": 11
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 7,
    "recordings": 7,
    "sessions": 5
   }
  },
  "MS-TCN": {
   "listed": {
    "units": 11,
    "recordings": 13,
    "sessions": 11
   },
   "fixed": {
    "units": 3,
    "recordings": 3,
    "sessions": 3
   },
   "new_misses": {
    "units": 8,
    "recordings": 8,
    "sessions": 6
   }
  },
  "ASRF": {
   "listed": {
    "units": 11,
    "recordings": 13,
    "sessions": 11
   },
   "fixed": {
    "units": 4,
    "recordings": 5,
    "sessions": 5
   },
   "new_misses": {
    "units": 20,
    "recordings": 19,
    "sessions": 12
   }
  }
 },
 "s1|calibration": {
  "Linear + B3": {
   "listed": {
    "units": 1,
    "recordings": 3,
    "sessions": 3
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 3,
    "recordings": 3,
    "sessions": 3
   }
  },
  "MS-TCN": {
   "listed": {
    "units": 1,
    "recordings": 3,
    "sessions": 3
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 7,
    "recordings": 9,
    "sessions": 9
   }
  },
  "ASRF": {
   "listed": {
    "units": 1,
    "recordings": 3,
    "sessions": 3
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 11,
    "recordings": 14,
    "sessions": 12
   }
  }
 },
 "s2|selection": {
  "Linear + B3": {
   "listed": {
    "units": 9,
    "recordings": 10,
    "sessions": 9
   },
   "fixed": {
    "units": 1,
    "recordings": 1,
    "sessions": 1
   },
   "new_misses": {
    "units": 5,
    "recordings": 5,
    "sessions": 5
   }
  },
  "MS-TCN": {
   "listed": {
    "units": 9,
    "recordings": 10,
    "sessions": 9
   },
   "fixed": {
    "units": 2,
    "recordings": 2,
    "sessions": 2
   },
   "new_misses": {
    "units": 12,
    "recordings": 14,
    "sessions": 10
   }
  },
  "ASRF": {
   "listed": {
    "units": 9,
    "recordings": 10,
    "sessions": 9
   },
   "fixed": {
    "units": 2,
    "recordings": 2,
    "sessions": 2
   },
   "new_misses": {
    "units": 19,
    "recordings": 18,
    "sessions": 12
   }
  }
 },
 "s2|calibration": {
  "Linear + B3": {
   "listed": {
    "units": 2,
    "recordings": 2,
    "sessions": 2
   },
   "fixed": {
    "units": 1,
    "recordings": 1,
    "sessions": 1
   },
   "new_misses": {
    "units": 1,
    "recordings": 1,
    "sessions": 1
   }
  },
  "MS-TCN": {
   "listed": {
    "units": 2,
    "recordings": 2,
    "sessions": 2
   },
   "fixed": {
    "units": 2,
    "recordings": 2,
    "sessions": 2
   },
   "new_misses": {
    "units": 11,
    "recordings": 17,
    "sessions": 14
   }
  },
  "ASRF": {
   "listed": {
    "units": 2,
    "recordings": 2,
    "sessions": 2
   },
   "fixed": {
    "units": 0,
    "recordings": 0,
    "sessions": 0
   },
   "new_misses": {
    "units": 12,
    "recordings": 15,
    "sessions": 11
   }
  }
 }
}
```

## ASRF decomposition

```
{
 "s0|selection": {
  "units": {
   "units": 502,
   "unmatched": {
    "units": 37,
    "recordings": 26,
    "sessions": 16
   },
   "edges": {
    "edges": 62,
    "proposed": 42,
    "merged": 10,
    "not_proposed": 20
   },
   "readings": {
    "source right, class branch wrong": {
     "units": 12,
     "recordings": 9,
     "sessions": 7
    },
    "class branch right, refinement wrong": {
     "units": 14,
     "recordings": 17,
     "sessions": 12
    },
    "edge not proposed": {
     "units": 18,
     "recordings": 15,
     "sessions": 13
    },
    "proposed but merged": {
     "units": 10,
     "recordings": 11,
     "sessions": 8
    }
   }
  },
  "natives": {
   "clips": 284,
   "wrong": {
    "clips": 2,
    "recordings": 2,
    "sessions": 2
   },
   "readings": {
    "source right, class branch wrong": {
     "clips": 1,
     "recordings": 1,
     "sessions": 1
    },
    "class branch right, refinement wrong": {
     "clips": 0,
     "recordings": 0,
     "sessions": 0
    }
   }
  }
 },
 "s0|calibration": {
  "units": {
   "units": 518,
   "unmatched": {
    "units": 15,
    "recordings": 17,
    "sessions": 13
   },
   "edges": {
    "edges": 26,
    "proposed": 19,
    "merged": 6,
    "not_proposed": 7
   },
   "readings": {
    "source right, class branch wrong": {
     "units": 8,
     "recordings": 12,
     "sessions": 10
    },
    "class branch right, refinement wrong": {
     "units": 4,
     "recordings": 4,
     "sessions": 4
    },
    "edge not proposed": {
     "units": 7,
     "recordings": 9,
     "sessions": 8
    },
    "proposed but merged": {
     "units": 5,
     "recordings": 5,
     "sessions": 3
    }
   }
  },
  "natives": {
   "clips": 237,
   "wrong": {
    "clips": 3,
    "recordings": 3,
    "sessions": 3
   },
   "readings": {
    "source right, class branch wrong": {
     "clips": 2,
     "recordings": 2,
     "sessions": 2
    },
    "class branch right, refinement wrong": {
     "clips": 0,
     "recordings": 0,
     "sessions": 0
    }
   }
  }
 },
 "s1|selection": {
  "units": {
   "units": 502,
   "unmatched": {
    "units": 32,
    "recordings": 27,
    "sessions": 14
   },
   "edges": {
    "edges": 54,
    "proposed": 43,
    "merged": 19,
    "not_proposed": 11
   },
   "readings": {
    "source right, class branch wrong": {
     "units": 11,
     "recordings": 10,
     "sessions": 8
    },
    "class branch right, refinement wrong": {
     "units": 14,
     "recordings": 16,
     "sessions": 12
    },
    "edge not proposed": {
     "units": 10,
     "recordings": 11,
     "sessions": 8
    },
    "proposed but merged": {
     "units": 18,
     "recordings": 17,
     "sessions": 11
    }
   }
  },
  "natives": {
   "clips": 284,
   "wrong": {
    "clips": 3,
    "recordings": 3,
    "sessions": 3
   },
   "readings": {
    "source right, class branch wrong": {
     "clips": 1,
     "recordings": 1,
     "sessions": 1
    },
    "class branch right, refinement wrong": {
     "clips": 1,
     "recordings": 1,
     "sessions": 1
    }
   }
  }
 },
 "s1|calibration": {
  "units": {
   "units": 518,
   "unmatched": {
    "units": 17,
    "recordings": 19,
    "sessions": 16
   },
   "edges": {
    "edges": 28,
    "proposed": 20,
    "merged": 6,
    "not_proposed": 8
   },
   "readings": {
    "source right, class branch wrong": {
     "units": 5,
     "recordings": 9,
     "sessions": 9
    },
    "class branch right, refinement wrong": {
     "units": 7,
     "recordings": 7,
     "sessions": 6
    },
    "edge not proposed": {
     "units": 8,
     "recordings": 10,
     "sessions": 10
    },
    "proposed but merged": {
     "units": 5,
     "recordings": 6,
     "sessions": 4
    }
   }
  },
  "natives": {
   "clips": 237,
   "wrong": {
    "clips": 4,
    "recordings": 4,
    "sessions": 3
   },
   "readings": {
    "source right, class branch wrong": {
     "clips": 3,
     "recordings": 3,
     "sessions": 2
    },
    "class branch right, refinement wrong": {
     "clips": 0,
     "recordings": 0,
     "sessions": 0
    }
   }
  }
 },
 "s2|selection": {
  "units": {
   "units": 502,
   "unmatched": {
    "units": 31,
    "recordings": 25,
    "sessions": 13
   },
   "edges": {
    "edges": 55,
    "proposed": 43,
    "merged": 19,
    "not_proposed": 12
   },
   "readings": {
    "source right, class branch wrong": {
     "units": 11,
     "recordings": 8,
     "sessions": 7
    },
    "class branch right, refinement wrong": {
     "units": 14,
     "recordings": 16,
     "sessions": 11
    },
    "edge not proposed": {
     "units": 10,
     "recordings": 11,
     "sessions": 9
    },
    "proposed but merged": {
     "units": 17,
     "recordings": 14,
     "sessions": 11
    }
   }
  },
  "natives": {
   "clips": 284,
   "wrong": {
    "clips": 2,
    "recordings": 2,
    "sessions": 2
   },
   "readings": {
    "source right, class branch wrong": {
     "clips": 1,
     "recordings": 1,
     "sessions": 1
    },
    "class branch right, refinement wrong": {
     "clips": 0,
     "recordings": 0,
     "sessions": 0
    }
   }
  }
 },
 "s2|calibration": {
  "units": {
   "units": 518,
   "unmatched": {
    "units": 21,
    "recordings": 22,
    "sessions": 15
   },
   "edges": {
    "edges": 34,
    "proposed": 28,
    "merged": 14,
    "not_proposed": 6
   },
   "readings": {
    "source right, class branch wrong": {
     "units": 6,
     "recordings": 7,
     "sessions": 5
    },
    "class branch right, refinement wrong": {
     "units": 8,
     "recordings": 11,
     "sessions": 10
    },
    "edge not proposed": {
     "units": 6,
     "recordings": 7,
     "sessions": 7
    },
    "proposed but merged": {
     "units": 12,
     "recordings": 15,
     "sessions": 11
    }
   }
  },
  "natives": {
   "clips": 237,
   "wrong": {
    "clips": 1,
    "recordings": 1,
    "sessions": 1
   },
   "readings": {
    "source right, class branch wrong": {
     "clips": 0,
     "recordings": 0,
     "sessions": 0
    },
    "class branch right, refinement wrong": {
     "clips": 0,
     "recordings": 0,
     "sessions": 0
    }
   }
  }
 }
}
```

## Overlap with Linear + B3

```
{
 "per_seed_half": {
  "s0|selection": {
   "ASRF": {
    "fixed": 7,
    "also_by_linear": 4,
    "share": 0.5714285714285714
   },
   "MS-TCN": {
    "fixed": 12,
    "also_by_linear": 6,
    "share": 0.5
   }
  },
  "s0|calibration": {
   "ASRF": {
    "fixed": 2,
    "also_by_linear": 1,
    "share": 0.5
   },
   "MS-TCN": {
    "fixed": 1,
    "also_by_linear": 1,
    "share": 1.0
   }
  },
  "s1|selection": {
   "ASRF": {
    "fixed": 7,
    "also_by_linear": 0,
    "share": 0.0
   },
   "MS-TCN": {
    "fixed": 7,
    "also_by_linear": 0,
    "share": 0.0
   }
  },
  "s1|calibration": {
   "ASRF": {
    "fixed": 2,
    "also_by_linear": 0,
    "share": 0.0
   },
   "MS-TCN": {
    "fixed": 2,
    "also_by_linear": 0,
    "share": 0.0
   }
  },
  "s2|selection": {
   "ASRF": {
    "fixed": 6,
    "also_by_linear": 2,
    "share": 0.3333333333333333
   },
   "MS-TCN": {
    "fixed": 5,
    "also_by_linear": 2,
    "share": 0.4
   }
  },
  "s2|calibration": {
   "ASRF": {
    "fixed": 2,
    "also_by_linear": 0,
    "share": 0.0
   },
   "MS-TCN": {
    "fixed": 6,
    "also_by_linear": 1,
    "share": 0.16666666666666666
   }
  }
 },
 "pooled": {
  "ASRF": {
   "fixed": 26,
   "also_by_linear": 7,
   "share": 0.2692307692307692
  },
  "MS-TCN": {
   "fixed": 33,
   "also_by_linear": 10,
   "share": 0.30303030303030304
  }
 }
}
```

## Short segments and X-Y-X middles

```
{
 "Linear + B3 s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 131,
   "wrong view": 0,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.9290780141843972,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 55,
   "wrong view": 0,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.8461538461538461,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 64,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.927536231884058,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "Linear + B3 s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 126,
   "wrong view": 2,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 63,
   "wrong view": 2,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 80,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "Linear argmax s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 135,
   "wrong view": 0,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9574468085106383,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 61,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9384615384615385,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 64,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.927536231884058,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "Linear argmax s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 124,
   "wrong view": 2,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 8,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 61,
   "wrong view": 2,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 77,
   "wrong view": 0,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9625,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "MS-TCN s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 130,
   "wrong view": 1,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.9219858156028369,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 56,
   "wrong view": 0,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.8615384615384616,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 64,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.927536231884058,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "MS-TCN s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 124,
   "wrong view": 2,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 61,
   "wrong view": 2,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "ASRF s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 128,
   "wrong view": 0,
   "no segment at IoU 0.5": 13,
   "taken by another reference": 0,
   "recall": 0.9078014184397163,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 52,
   "wrong view": 0,
   "no segment at IoU 0.5": 13,
   "taken by another reference": 0,
   "recall": 0.8,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 7,
   "missed_sessions": 7
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "ASRF s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 124,
   "wrong view": 1,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 9,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 62,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 77,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9625,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "ASRF frame branch s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 129,
   "wrong view": 0,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.9148936170212766,
   "missed_recordings": 8,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 53,
   "wrong view": 0,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.8153846153846154,
   "missed_recordings": 8,
   "missed_sessions": 7
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "ASRF frame branch s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 123,
   "wrong view": 1,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.917910447761194,
   "missed_recordings": 10,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 61,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 77,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9625,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "Linear + B3 s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 129,
   "wrong view": 0,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.9148936170212766,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 53,
   "wrong view": 0,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.8153846153846154,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 64,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.927536231884058,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "Linear + B3 s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 127,
   "wrong view": 3,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9477611940298507,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 63,
   "wrong view": 3,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "Linear argmax s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 134,
   "wrong view": 0,
   "no segment at IoU 0.5": 7,
   "taken by another reference": 0,
   "recall": 0.950354609929078,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 60,
   "wrong view": 0,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9230769230769231,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 65,
   "wrong view": 1,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9420289855072463,
   "missed_recordings": 5,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "Linear argmax s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 126,
   "wrong view": 3,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 63,
   "wrong view": 3,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "MS-TCN s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 131,
   "wrong view": 1,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.9290780141843972,
   "missed_recordings": 7,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 55,
   "wrong view": 1,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.8461538461538461,
   "missed_recordings": 7,
   "missed_sessions": 7
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "MS-TCN s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 126,
   "wrong view": 2,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 8,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 62,
   "wrong view": 2,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "ASRF s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 126,
   "wrong view": 1,
   "no segment at IoU 0.5": 14,
   "taken by another reference": 0,
   "recall": 0.8936170212765957,
   "missed_recordings": 10,
   "missed_sessions": 10
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 52,
   "wrong view": 1,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.8,
   "missed_recordings": 8,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "ASRF s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 127,
   "wrong view": 1,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9477611940298507,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "ASRF frame branch s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 127,
   "wrong view": 2,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.900709219858156,
   "missed_recordings": 9,
   "missed_sessions": 9
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 52,
   "wrong view": 2,
   "no segment at IoU 0.5": 11,
   "taken by another reference": 0,
   "recall": 0.8,
   "missed_recordings": 8,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "ASRF frame branch s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 129,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9626865671641791,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 65,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9701492537313433,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "Linear + B3 s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 135,
   "wrong view": 0,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9574468085106383,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 59,
   "wrong view": 0,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9076923076923077,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 66,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9565217391304348,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "Linear + B3 s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 126,
   "wrong view": 3,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 64,
   "wrong view": 3,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9552238805970149,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "Linear argmax s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 138,
   "wrong view": 0,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9787234042553191,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 62,
   "wrong view": 0,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9538461538461539,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 66,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9565217391304348,
   "missed_recordings": 4,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "Linear argmax s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 125,
   "wrong view": 3,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9328358208955224,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 62,
   "wrong view": 3,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 4,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 77,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9625,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "MS-TCN s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 128,
   "wrong view": 1,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.9078014184397163,
   "missed_recordings": 10,
   "missed_sessions": 9
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 53,
   "wrong view": 1,
   "no segment at IoU 0.5": 11,
   "taken by another reference": 0,
   "recall": 0.8153846153846154,
   "missed_recordings": 9,
   "missed_sessions": 9
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 7,
   "missed_sessions": 7
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 2,
   "wrong view": 0,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.2857142857142857,
   "missed_recordings": 6,
   "missed_sessions": 6
  }
 },
 "MS-TCN s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 126,
   "wrong view": 1,
   "no segment at IoU 0.5": 7,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 8,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 63,
   "wrong view": 1,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 4,
   "missed_sessions": 3
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 77,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9625,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 1
  }
 },
 "ASRF s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 124,
   "wrong view": 0,
   "no segment at IoU 0.5": 17,
   "taken by another reference": 0,
   "recall": 0.8794326241134752,
   "missed_recordings": 13,
   "missed_sessions": 11
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 51,
   "wrong view": 0,
   "no segment at IoU 0.5": 14,
   "taken by another reference": 0,
   "recall": 0.7846153846153846,
   "missed_recordings": 10,
   "missed_sessions": 9
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 62,
   "wrong view": 1,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.8985507246376812,
   "missed_recordings": 8,
   "missed_sessions": 8
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 2,
   "wrong view": 0,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.2857142857142857,
   "missed_recordings": 6,
   "missed_sessions": 6
  }
 },
 "ASRF s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 122,
   "wrong view": 2,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 11,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 60,
   "wrong view": 2,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.8955223880597015,
   "missed_recordings": 7,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 2,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 1
  }
 },
 "ASRF frame branch s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 128,
   "wrong view": 0,
   "no segment at IoU 0.5": 13,
   "taken by another reference": 0,
   "recall": 0.9078014184397163,
   "missed_recordings": 10,
   "missed_sessions": 9
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 53,
   "wrong view": 0,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.8153846153846154,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 62,
   "wrong view": 1,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.8985507246376812,
   "missed_recordings": 8,
   "missed_sessions": 8
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 1,
   "wrong view": 0,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.14285714285714285,
   "missed_recordings": 7,
   "missed_sessions": 7
  }
 },
 "ASRF frame branch s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 123,
   "wrong view": 2,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.917910447761194,
   "missed_recordings": 10,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 61,
   "wrong view": 2,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 6,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 2,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 1
  }
 },
 "frozen Ordered + B3 s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 130,
   "wrong view": 2,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.9219858156028369,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 54,
   "wrong view": 2,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.8307692307692308,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 66,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9565217391304348,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "frozen Ordered + B3 s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 128,
   "wrong view": 1,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9552238805970149,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 66,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9850746268656716,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 80,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "frozen Ordered argmax s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 135,
   "wrong view": 2,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9574468085106383,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 60,
   "wrong view": 2,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9230769230769231,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 66,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9565217391304348,
   "missed_recordings": 4,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "frozen Ordered argmax s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 127,
   "wrong view": 1,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9477611940298507,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 65,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9701492537313433,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 80,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "frozen Ordered + B3 s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 134,
   "wrong view": 0,
   "no segment at IoU 0.5": 7,
   "taken by another reference": 0,
   "recall": 0.950354609929078,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 58,
   "wrong view": 0,
   "no segment at IoU 0.5": 7,
   "taken by another reference": 0,
   "recall": 0.8923076923076924,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 64,
   "wrong view": 1,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.927536231884058,
   "missed_recordings": 6,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "frozen Ordered + B3 s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 129,
   "wrong view": 2,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9626865671641791,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 65,
   "wrong view": 2,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9701492537313433,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 80,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "frozen Ordered argmax s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 139,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9858156028368794,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 64,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9846153846153847,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 65,
   "wrong view": 1,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.9420289855072463,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "frozen Ordered argmax s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 127,
   "wrong view": 3,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9477611940298507,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 64,
   "wrong view": 3,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9552238805970149,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "frozen Ordered + B3 s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 136,
   "wrong view": 0,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9645390070921985,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 60,
   "wrong view": 0,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9230769230769231,
   "missed_recordings": 5,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 66,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9565217391304348,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "frozen Ordered + B3 s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 127,
   "wrong view": 2,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9477611940298507,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 65,
   "wrong view": 2,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9701492537313433,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 80,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "frozen Ordered argmax s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 139,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9858156028368794,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 64,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9846153846153847,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 67,
   "wrong view": 1,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.9710144927536232,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "frozen Ordered argmax s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 126,
   "wrong view": 3,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9402985074626866,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 64,
   "wrong view": 3,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9552238805970149,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "E1 + B3|selection": {
  "short segments": {
   "segments": 141,
   "matched": 129,
   "wrong view": 6,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.9148936170212766,
   "missed_recordings": 10,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 56,
   "wrong view": 3,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.8615384615384616,
   "missed_recordings": 9,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 63,
   "wrong view": 2,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9130434782608695,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "E1 + B3|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 125,
   "wrong view": 5,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9328358208955224,
   "missed_recordings": 8,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 59,
   "wrong view": 4,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.8805970149253731,
   "missed_recordings": 7,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 80,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "Original FACT s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 127,
   "wrong view": 5,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.900709219858156,
   "missed_recordings": 11,
   "missed_sessions": 9
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 54,
   "wrong view": 2,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.8307692307692308,
   "missed_recordings": 10,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 60,
   "wrong view": 1,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.8695652173913043,
   "missed_recordings": 10,
   "missed_sessions": 10
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 1,
   "wrong view": 0,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.14285714285714285,
   "missed_recordings": 7,
   "missed_sessions": 7
  }
 },
 "Original FACT s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 124,
   "wrong view": 5,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9253731343283582,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 58,
   "wrong view": 4,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.8656716417910447,
   "missed_recordings": 5,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 1,
   "missed_sessions": 1
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "Original FACT s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 127,
   "wrong view": 6,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.900709219858156,
   "missed_recordings": 11,
   "missed_sessions": 9
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 56,
   "wrong view": 2,
   "no segment at IoU 0.5": 7,
   "taken by another reference": 0,
   "recall": 0.8615384615384616,
   "missed_recordings": 8,
   "missed_sessions": 7
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 60,
   "wrong view": 2,
   "no segment at IoU 0.5": 7,
   "taken by another reference": 0,
   "recall": 0.8695652173913043,
   "missed_recordings": 9,
   "missed_sessions": 9
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 4,
   "missed_sessions": 4
  }
 },
 "Original FACT s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 123,
   "wrong view": 6,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.917910447761194,
   "missed_recordings": 8,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 57,
   "wrong view": 5,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.8507462686567164,
   "missed_recordings": 7,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 77,
   "wrong view": 1,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9625,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 5,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.7142857142857143,
   "missed_recordings": 2,
   "missed_sessions": 2
  }
 },
 "Original FACT s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 128,
   "wrong view": 5,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.9078014184397163,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 55,
   "wrong view": 2,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.8461538461538461,
   "missed_recordings": 8,
   "missed_sessions": 7
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 59,
   "wrong view": 1,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.855072463768116,
   "missed_recordings": 12,
   "missed_sessions": 12
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 4,
   "missed_sessions": 4
  }
 },
 "Original FACT s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 125,
   "wrong view": 4,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.9328358208955224,
   "missed_recordings": 7,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 59,
   "wrong view": 4,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.8805970149253731,
   "missed_recordings": 6,
   "missed_sessions": 4
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 78,
   "wrong view": 0,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.975,
   "missed_recordings": 2,
   "missed_sessions": 2
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 6,
   "wrong view": 0,
   "no segment at IoU 0.5": 1,
   "taken by another reference": 0,
   "recall": 0.8571428571428571,
   "missed_recordings": 1,
   "missed_sessions": 1
  }
 },
 "paired ASRF s0|selection": {
  "short segments": {
   "segments": 141,
   "matched": 125,
   "wrong view": 6,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.8865248226950354,
   "missed_recordings": 14,
   "missed_sessions": 9
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 53,
   "wrong view": 3,
   "no segment at IoU 0.5": 9,
   "taken by another reference": 0,
   "recall": 0.8153846153846154,
   "missed_recordings": 12,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 60,
   "wrong view": 4,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.8695652173913043,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 4,
   "wrong view": 0,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.5714285714285714,
   "missed_recordings": 4,
   "missed_sessions": 4
  }
 },
 "paired ASRF s0|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 127,
   "wrong view": 5,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9477611940298507,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 61,
   "wrong view": 4,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "paired ASRF s1|selection": {
  "short segments": {
   "segments": 141,
   "matched": 123,
   "wrong view": 6,
   "no segment at IoU 0.5": 12,
   "taken by another reference": 0,
   "recall": 0.8723404255319149,
   "missed_recordings": 15,
   "missed_sessions": 10
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 51,
   "wrong view": 3,
   "no segment at IoU 0.5": 11,
   "taken by another reference": 0,
   "recall": 0.7846153846153846,
   "missed_recordings": 13,
   "missed_sessions": 9
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 60,
   "wrong view": 3,
   "no segment at IoU 0.5": 6,
   "taken by another reference": 0,
   "recall": 0.8695652173913043,
   "missed_recordings": 10,
   "missed_sessions": 9
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 3,
   "wrong view": 0,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.42857142857142855,
   "missed_recordings": 5,
   "missed_sessions": 5
  }
 },
 "paired ASRF s1|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 128,
   "wrong view": 4,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9552238805970149,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 61,
   "wrong view": 4,
   "no segment at IoU 0.5": 2,
   "taken by another reference": 0,
   "recall": 0.9104477611940298,
   "missed_recordings": 6,
   "missed_sessions": 5
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 },
 "paired ASRF s2|selection": {
  "short segments": {
   "segments": 141,
   "matched": 125,
   "wrong view": 6,
   "no segment at IoU 0.5": 10,
   "taken by another reference": 0,
   "recall": 0.8865248226950354,
   "missed_recordings": 14,
   "missed_sessions": 10
  },
  "short A4C or A5C segments": {
   "segments": 65,
   "matched": 54,
   "wrong view": 3,
   "no segment at IoU 0.5": 8,
   "taken by another reference": 0,
   "recall": 0.8307692307692308,
   "missed_recordings": 11,
   "missed_sessions": 8
  },
  "X-Y-X middles": {
   "segments": 69,
   "matched": 60,
   "wrong view": 4,
   "no segment at IoU 0.5": 5,
   "taken by another reference": 0,
   "recall": 0.8695652173913043,
   "missed_recordings": 9,
   "missed_sessions": 8
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 4,
   "wrong view": 0,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.5714285714285714,
   "missed_recordings": 4,
   "missed_sessions": 4
  }
 },
 "paired ASRF s2|calibration": {
  "short segments": {
   "segments": 134,
   "matched": 125,
   "wrong view": 5,
   "no segment at IoU 0.5": 4,
   "taken by another reference": 0,
   "recall": 0.9328358208955224,
   "missed_recordings": 9,
   "missed_sessions": 7
  },
  "short A4C or A5C segments": {
   "segments": 67,
   "matched": 60,
   "wrong view": 4,
   "no segment at IoU 0.5": 3,
   "taken by another reference": 0,
   "recall": 0.8955223880597015,
   "missed_recordings": 7,
   "missed_sessions": 6
  },
  "X-Y-X middles": {
   "segments": 80,
   "matched": 79,
   "wrong view": 1,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 0.9875,
   "missed_recordings": 3,
   "missed_sessions": 3
  },
  "A4C/A5C X-Y-X middles": {
   "segments": 7,
   "matched": 7,
   "wrong view": 0,
   "no segment at IoU 0.5": 0,
   "taken by another reference": 0,
   "recall": 1.0,
   "missed_recordings": 0,
   "missed_sessions": 0
  }
 }
}
```

## Native errors

```
{
 "Linear + B3|selection": {
  "PLAX->PSAX": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  },
  "A4C->A5C": {
   "clips_over_seeds": 2,
   "recordings": 2,
   "sessions": 2
  }
 },
 "MS-TCN|selection": {
  "PLAX->PSAX": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  },
  "A5C->A4C": {
   "clips_over_seeds": 3,
   "recordings": 3,
   "sessions": 3
  },
  "A4C->A5C": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  }
 },
 "ASRF|selection": {
  "PLAX->PSAX": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  },
  "A5C->A4C": {
   "clips_over_seeds": 2,
   "recordings": 2,
   "sessions": 2
  },
  "A4C->A5C": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  }
 },
 "frozen Ordered + B3|selection": {
  "PLAX->PSAX": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  },
  "A4C->A5C": {
   "clips_over_seeds": 3,
   "recordings": 3,
   "sessions": 3
  }
 },
 "Linear + B3|calibration": {
  "A4C->A5C": {
   "clips_over_seeds": 3,
   "recordings": 3,
   "sessions": 3
  }
 },
 "MS-TCN|calibration": {
  "A5C->A4C": {
   "clips_over_seeds": 4,
   "recordings": 4,
   "sessions": 4
  },
  "A4C->A5C": {
   "clips_over_seeds": 1,
   "recordings": 1,
   "sessions": 1
  }
 },
 "ASRF|calibration": {
  "A5C->A4C": {
   "clips_over_seeds": 2,
   "recordings": 2,
   "sessions": 2
  },
  "A4C->A5C": {
   "clips_over_seeds": 3,
   "recordings": 3,
   "sessions": 2
  }
 },
 "frozen Ordered + B3|calibration": {
  "A4C->A5C": {
   "clips_over_seeds": 2,
   "recordings": 2,
   "sessions": 2
  }
 }
}
```

## Parity and counts

```
{
 "s0|selection": {
  "windows": 7666,
  "max_abs": 8.10816392871061e-06,
  "argmax_equal": true
 },
 "s0|calibration": {
  "windows": 7021,
  "max_abs": 6.429890987647013e-06,
  "argmax_equal": true
 },
 "s1|selection": {
  "windows": 7666,
  "max_abs": 7.961907670051005e-06,
  "argmax_equal": true
 },
 "s1|calibration": {
  "windows": 7021,
  "max_abs": 7.40092592721453e-06,
  "argmax_equal": true
 },
 "s2|selection": {
  "windows": 7666,
  "max_abs": 6.5652288725459584e-06,
  "argmax_equal": true
 },
 "s2|calibration": {
  "windows": 7021,
  "max_abs": 7.827656521186555e-06,
  "argmax_equal": true
 }
}
```

```
{
 "selection": {
  "sequences": 541,
  "samples": 37287
 },
 "calibration": {
  "sequences": 472,
  "samples": 34216
 }
}
```

## Cost

```
{
 "ledger_hours_before_evaluation": 3.7469408437940808,
 "evaluation_seconds": 1373.7355830669403
}
```

## Decision

Applied: tradeoff, no_gain, end

```
{
 "rows": {
  "system_gain": {
   "applies": false,
   "reading": "ASRF - (Linear + B3) and ASRF - (frozen Ordered + B3) are both consistent positive in multi segment F1@50 in both halves, and ASRF has no constraint failure in any seed and half -> 'a system gain of this whole temporal integration'; never attributed to the boundary loss alone and never called a new architecture"
  },
  "selection_only": {
   "applies": false,
   "reading": "ASRF - (frozen Ordered + B3) segment F1@50 is consistent positive on selection but not on calibration, and its calibration native macro-F1 difference is not consistent positive -> 'local progress; the overall goal is not met'; this configuration ends without visual fine-tuning or a loss remedy"
  },
  "linear_explains": {
   "applies": false,
   "share": 0.2692307692307692,
   "reading": "ASRF - (Linear + B3) segment F1@50 is consistent positive in neither half, and more than half of the units and native clips ASRF fixes against frozen Ordered + B3 (pooled over seeds and halves, each seed against its own Linear + B3) are also fixed by Linear + B3 -> 'class re-calibration can explain part of the result; no temporal gain detected'; not called a temporal mechanism"
  },
  "tradeoff": {
   "applies": true,
   "where": [
    "ASRF|calibration",
    "ASRF|selection",
    "MS-TCN|calibration",
    "MS-TCN|selection"
   ],
   "reading": "for ASRF or MS-TCN in a half: it fixes at least one unit or native clip against frozen Ordered + B3 (pooled over seeds), it newly misses at least one native clip or has a consistent positive difference in native label switches or same-view join cuts against frozen Ordered + B3, and its segment F1@50 difference against frozen Ordered + B3 is not consistent positive -> the trade-off is reported and this configuration ends; no source-logit residual or gate is added and no official threshold is retuned"
  },
  "no_gain": {
   "applies": true,
   "mstcn_leads_in": [],
   "reading": "ASRF - (Linear + B3) and ASRF - (frozen Ordered + B3) segment F1@50 are consistent positive in neither half -> this video-feature temporal recipe ends; temporal models in general are not rejected; a half where ASRF - MS-TCN is consistent negative is reported as MS-TCN leading, without changing the primary candidate or searching"
  },
  "end": {
   "applies": true
  }
 },
 "applied": [
  "tradeoff",
  "no_gain",
  "end"
 ]
}
```
