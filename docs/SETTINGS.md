# Environment Variables (`SETTINGS.md`)

All runtime environment variables recognized by `tinymodel` are registered here (per `AGENTS.md` §6). Keep the list minimal.

| Variable | Default | Purpose | Status |
|---|---|---|---|
| `TM_FORCE_ISA` | *(unset — auto-detect widest supported ISA)* | Force CPU kernel dispatch to a specific ISA tier (`generic`, `neon`, `dotprod`, `i8mm`, `avx2`, `avx512`) for parity testing and benchmarking. | Stable |
| `TM_VK_DEVICE` | *(unset — best device: discrete > integrated > virtual > cpu)* | Select the Vulkan device for `tm-vulkan` by loader index (`1`) or case-insensitive name substring (`RX 9060`, `llvmpipe`). | Experimental |
| `TM_GLSLC` | *(unset — `$VULKAN_SDK/bin/glslc`, then `glslc` on `PATH`, else committed `crates/tm-vulkan/spv/*.spv`)* | **Build-time** path of the GLSL→SPIR-V compiler used by `tm-vulkan/build.rs`. | Experimental |
