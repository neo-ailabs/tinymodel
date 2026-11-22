# Environment Variables (`SETTINGS.md`)

All runtime environment variables recognized by `tinymodel` are registered here (per `AGENTS.md` §6). Keep the list minimal.

| Variable | Default | Purpose | Status |
|---|---|---|---|
| `TM_FORCE_ISA` | *(unset — auto-detect widest supported ISA)* | Force CPU kernel dispatch to a specific ISA tier (`generic`, `neon`, `dotprod`, `i8mm`, `avx2`, `avx512`) for parity testing and benchmarking. | Stable |
| `TM_VK_DEVICE` | *(unset — best device: discrete > integrated > virtual > cpu)* | Select the Vulkan device for `tm-vulkan` by loader index (`1`) or case-insensitive name substring (`radv`, `UHD`, `llvmpipe`). | Experimental |
| `TM_VK_PREFILL` | `auto` | Select batched-prefill execution path in `tm-vulkan` (`auto`, `f16` for `16×16×16` FP16 Cooperative Matrix + FlashAttention, `i8` for `16×16×16` INT8 Cooperative Matrix + `Q8_1` activation quant, `fallback` for portable LDS/subgroup kernels). | Experimental |
| `TM_GLSLC` | *(unset — `$VULKAN_SDK/bin/glslc`, then `glslc` on `PATH`, else committed `crates/tm-vulkan/spv/*.spv`)* | **Build-time** path of the GLSL→SPIR-V compiler used by `tm-vulkan/build.rs`. | Experimental |
| `TM_INKLING2_RAM_MIB` | *(unset — auto-planned from `probe_system_memory` with `2 GiB` OS/state headroom)* | Override host anonymous-RAM budget (in MiB) reserved for loop-prioritized MoE expert layers in `Inkling2Model` (`0` forces all non-VRAM expert layers onto `O_DIRECT` NVMe streaming). | Experimental |
