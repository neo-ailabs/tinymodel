# Third-Party Notices

This repository contains code ported from or directly inspired by the following open-source projects. Individual source files retain the original copyright notices and upstream commit hashes in their file headers.

## ggml / llama.cpp

- **License**: MIT License
- **Copyright**: Copyright (c) 2023-2026 The ggml authors
- **Upstream**: https://github.com/ggml-org/llama.cpp
- **Pinned commit**: `3cf03257f219afbe7334045ff7c6a06ac68c627d` (via Strata) and `b001d7b` (M0 baseline)
- **Used in**: Quantization block layouts (`BlockQ4_0`, `BlockQ8_0`) and ARM NEON / dotprod / i8mm vector dot product algorithms in `crates/tm-core` and `crates/tm-cpu`.

## Strata

- **License**: MIT License
- **Copyright**: Copyright (c) 2026 Niko1221 and the Strata contributors
- **Upstream**: https://github.com/Niko1221/Strata
- **Used in**: Runtime ISA probe design (`isa.rs`), kernel registry dispatch (`registry.rs`), epoch-CAS thread pool design (`pool.rs`), and power-of-two scale parity testing methodology.
