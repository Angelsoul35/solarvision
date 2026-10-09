# Datasets — real solar-cell images (ELPV)

## `elpv/<class>/*.png` — ELPV electroluminescence defect dataset

A **real**, public dataset of electroluminescence (EL) images of photovoltaic
solar cells, used to validate the classical-CV approach on real imagery (not only
on the synthetic generator).

| | |
|---|---|
| **Origin** | ELPV dataset — Buerhop-Lutz et al., *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery* (mirrored as `mjphayes/elpv-dataset` on Hugging Face) |
| **Task here** | binary **functional vs defective** (the standard ELPV benchmark) |
| **Images here** | 45 functional + 45 defective (90 total), 300×300 grayscale PNG, a balanced subset |
| **Retrieved** | 2026-10-09, via the Hugging Face datasets-server |
| **License** | Research use (per the ELPV dataset terms) |

The original ELPV labels are a defect *probability* (functional, possibly, likely,
certainly defective); here the three defective levels are grouped into one
`defective` class for the standard binary task. The images are **vendored**
(committed) so the benchmark and its tests run offline and deterministically;
`solarvision.real_data.fetch_elpv()` can pull more.

### What the real benchmark shows

`python -m solarvision.real_benchmark` extracts the **same** classical features used
on synthetic panels and trains on these real EL cells. Result: ~0.88 accuracy and
~0.89–0.94 ROC-AUC (binary, chance 0.50), with a well-fit logistic model. The
hand-defined features — dark-region coverage, crack-like elongation, local contrast
and gradient statistics — carry over directly to real electroluminescence imagery,
where a defect shows as a darker, often cracked region, exactly what they measure.
Every verdict still names the features that drove it.
