# Environment Variables (`SETTINGS.md`)

All runtime environment variables recognized by `tinymodel` are registered here (per `AGENTS.md` §6). Keep the list minimal.

| Variable | Default | Purpose | Status |
|---|---|---|---|
| `TM_FORCE_ISA` | *(unset — auto-detect widest supported ISA)* | Force CPU kernel dispatch to a specific ISA tier (`generic`, `neon`, `dotprod`, `i8mm`, `avx2`, `avx512`) for parity testing and benchmarking. | Stable |
