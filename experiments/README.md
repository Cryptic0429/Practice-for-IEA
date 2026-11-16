# Experimental Protocols & Evaluation Records

This directory preserves frozen training/evaluation protocols, training logs, visual inspection observations, and negative/replication-failure records across project iterations.

- All retained files verified by `reports/mainline-technical-20261113/generated/evidence.json` and `training-protocol.json` are kept byte-exact so `make verify` passes cleanly.
- See [docs/EVALUATION.md](../docs/EVALUATION.md) for a structured summary of ablations, benchmark rules, and known limitations.
- `comparison-studio-20261110/resource-ledger.json` is a historical experiment artifact; runtime uploads and sessions remain isolated and excluded via `.gitignore`.
