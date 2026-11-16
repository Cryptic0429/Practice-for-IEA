# Mainline Technical Report (`mainline-technical-20261113`)

Build the PDF (requires Tectonic with XeTeX, Heiti SC, and Arial fonts):

```bash
make report
make clean-report
```

- Vector architecture and lineage diagrams live in `figures/` (`lineage.tex`, `runtime.tex`) as TikZ sources.
- `generated/evidence.json` lists the 669 SHA256-verified source files checked by `make verify`.
