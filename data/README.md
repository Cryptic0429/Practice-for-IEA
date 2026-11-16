# Dataset Metadata & Benchmark Definitions

This directory retains structured JSON case definitions, admission records, and bilingual evaluation splits for `calibrated_sft`, `r1`, `r15`, and `r21_head`.

- Raw source photographs, target candidate renders, and large feature caches are excluded from Git via `.gitignore` (see [Reproducibility & Weight Setup](../docs/REPRODUCIBILITY.md)).
- Subdirectories under `data/training/` are kept byte-exact to preserve SHA256 verification in `make verify`.
- Bilingual translations and synthetic noise conditions are tracked by original image group rather than counted as independent photographs.
