# Experimental Protocols & Evaluation Records

This directory is the project's audit archive: original lab-notebook records (written in Chinese) preserving frozen training/evaluation protocols, training logs, visual inspection observations, and negative/replication-failure results across protocol iterations. Directory names carry internal version numbers; see the [experiment record index](../docs/MODEL_VERSIONS.md#5-experiment-record-index) for what each record documents.

- All retained files verified by `reports/mainline-technical-20261113/generated/evidence.json` and `training-protocol.json` are kept byte-exact so `make verify` passes cleanly.
- See [docs/EVALUATION.md](../docs/EVALUATION.md) for the English summary of ablations, benchmark rules, and known limitations.
- `comparison-studio-20261110/resource-ledger.json` is a historical experiment artifact; runtime uploads and sessions remain isolated and excluded via `.gitignore`.
