# Label mapping notes (family5_v1)

Raw EV9V codes are preserved verbatim. The dataset card and the STFM README disagree on several
English expansions; we do **not** resolve this silently. Frozen mapping lives in `configs/labels.yaml`.

| Raw code | Dataset card | STFM README | family5_v1 | Note |
|---|---|---|---|---|
| PLHLA | Parasternal Long Axis View | Parasternal Long Axis | PLAX | provisional |
| PASA | Parasternal Short Axis View | Parasternal Short Axis (Aortic Valve) | PSAX | level unresolved |
| PMASA | PSAX at Apical Level | PSAX (Mitral Valve) | PSAX | **conflict** |
| PMVLSA | PSAX at Mitral Valve Level | PSAX (Papillary Muscle) | PSAX | **conflict** |
| PPMLSA | PSAX at Papillary Muscle Level | PSAX (LV) | PSAX | level unresolved |
| PMPALA | Pulmonary Main Pulmonary Artery Long Axis | Parasternal Long Axis (Apical) | **excluded** | **conflict** |
| A4C | Apical 4-Chamber | Apical 4-Chamber | A4C | agree |
| A5C | Apical 5-Chamber | Apical 5-Chamber | A5C | agree |
| SC4C | Subcostal 4-Chamber | Subcostal 4-Chamber | SC4C | agree |

Class index (family5): PLAX 0, PSAX 1, A4C 2, A5C 3, SC4C 4. Raw9 index follows STFM `INDEXOFLABEL`.
Open action: ask the EV9V authors (or a clinician reviewer) to confirm the PSAX-level codes and PMPALA.
Any change becomes `family5_v2` and re-runs from the feature cache.
