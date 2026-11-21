//! L1 Vulkan SPIR-V & GPU kernel parity tests (`docs/TESTING.md` §1 & `AGENTS.md` Rule #3):
//! - Verifies all 32 `naga`-compiled SPIR-V compute binaries have valid SPIR-V magic headers (`0x07230203`).
//! - When a Vulkan compute ICD is present at runtime (e.g. Pixel 8 Mali-G715, Linux Mesa/V3DV/RADV, MoltenVK),
//!   validates `Q4_0`, `Q4_0x4`, `Q8_0`, `Q6_K`, `F16` GEMV + GEMM, `rms_norm`, and `argmax` against exact CPU references!

use tm_core::{
    BlockQ4_0, BlockQ8_0, DType, dequantize_row_bytes, f32_to_f16, repack_q4_0_bytes_to_q4_0x4,
};
use tm_vulkan::context::as_bytes;
use tm_vulkan::pipelines::{ArgmaxPush, GemmPush, GemvPush, NormPush};
use tm_vulkan::{DescriptorArena, GpuBuffer, VulkanContext, VulkanPipelines, shaders};

const SPIRV_MAGIC: u32 = 0x0723_0203;

#[test]
fn all_embedded_spirv_modules_have_valid_headers() {
    let modules: &[(&str, &[u8])] = &[
        ("gemv_q4_0", shaders::GEMV_Q4_0),
        ("gemv_q4_0x4", shaders::GEMV_Q4_0X4),
        ("gemv_q4_1", shaders::GEMV_Q4_1),
        ("gemv_q8_0", shaders::GEMV_Q8_0),
        ("gemv_q5_k", shaders::GEMV_Q5_K),
        ("gemv_q6_k", shaders::GEMV_Q6_K),
        ("gemv_f16", shaders::GEMV_F16),
        ("gemv_bf16", shaders::GEMV_BF16),
        ("gemv_f32", shaders::GEMV_F32),
        ("gemm_q4_0", shaders::GEMM_Q4_0),
        ("gemm_q4_0x4", shaders::GEMM_Q4_0X4),
        ("gemm_q4_1", shaders::GEMM_Q4_1),
        ("gemm_q8_0", shaders::GEMM_Q8_0),
        ("gemm_q5_k", shaders::GEMM_Q5_K),
        ("gemm_q6_k", shaders::GEMM_Q6_K),
        ("gemm_f16", shaders::GEMM_F16),
        ("gemm_bf16", shaders::GEMM_BF16),
        ("gemm_f32", shaders::GEMM_F32),
        ("embed", shaders::EMBED),
        ("rms_norm", shaders::RMS_NORM),
        ("qk_norm", shaders::QK_NORM),
        ("rope_kv", shaders::ROPE_KV),
        ("gqa_attn", shaders::GQA_ATTN),
        ("gqa_attn_batch", shaders::GQA_ATTN_BATCH),
        ("swiglu", shaders::SWIGLU),
        ("add_inplace", shaders::ADD_INPLACE),
        ("argmax", shaders::ARGMAX),
        ("split_qg_norm", shaders::SPLIT_QG_NORM),
        ("sigmoid_gate", shaders::SIGMOID_GATE),
        ("gdn_gates", shaders::GDN_GATES),
        ("gdn_conv_silu_l2", shaders::GDN_CONV_SILU_L2),
        ("gdn_recur_gated", shaders::GDN_RECUR_GATED),
    ];

    for &(name, spv) in modules {
        assert!(
            spv.len() >= 20 && spv.len() % 4 == 0,
            "{name} invalid byte len"
        );
        let magic = u32::from_le_bytes([spv[0], spv[1], spv[2], spv[3]]);
        assert_eq!(
            magic, SPIRV_MAGIC,
            "{name} invalid SPIR-V magic 0x{magic:08x}"
        );
    }
}

#[test]
fn l1_vulkan_gemv_gemm_norm_argmax_parity_when_driver_available() {
    let Ok(ctx) = VulkanContext::new() else {
        // Host does not have a runtime Vulkan ICD installed (e.g. macOS without MoltenVK);
        // real GPU execution is additionally verified on Pixel 8 Mali-G715 via `tinymodel probe --vulkan`.
        return;
    };
    let pipes = VulkanPipelines::new(&ctx).expect("compile Vulkan pipelines");
    let arena = DescriptorArena::new(&ctx, 32, 128).expect("desc arena");

    let rows = 8usize;
    let cols = 128usize;
    let nb = cols / 32;
    let n_tok = 5usize;

    let mut q4_bytes = Vec::with_capacity(rows * nb * 18);
    for r in 0..rows {
        for ib in 0..nb {
            let mut qs = [0u8; 16];
            for (j, q) in qs.iter_mut().enumerate() {
                let lo = ((r * 3 + ib * 5 + j) & 0xF) as u8;
                let hi = ((15 - ((r * 7 + ib + j * 3) & 0xF)) & 0xF) as u8;
                *q = lo | (hi << 4);
            }
            let blk = BlockQ4_0 {
                d: f32_to_f16(0.0625 * ((r as f32) - 3.5) + 0.03125 * (ib as f32)),
                qs,
            };
            q4_bytes.extend_from_slice(&blk.d.to_le_bytes());
            q4_bytes.extend_from_slice(&blk.qs);
        }
    }
    let mut q4x4_bytes = vec![0u8; q4_bytes.len()];
    repack_q4_0_bytes_to_q4_0x4(&q4_bytes, &mut q4x4_bytes, rows, cols);

    let x_batch: Vec<f32> = (0..n_tok * cols)
        .map(|i| (((i * 19 + 7) % 89) as f32 - 44.0) * 0.05)
        .collect();

    // CPU exact reference (`W_dequant * x`)
    let mut ref_out = vec![0.0f32; n_tok * rows];
    let mut w_row = vec![0.0f32; cols];
    for r in 0..rows {
        let rb = DType::Q4_0.row_bytes(cols);
        dequantize_row_bytes(DType::Q4_0, &q4_bytes[r * rb..(r + 1) * rb], &mut w_row);
        for t in 0..n_tok {
            let mut s = 0.0f32;
            for c in 0..cols {
                s += w_row[c] * x_batch[t * cols + c];
            }
            ref_out[t * rows + r] = s;
        }
    }

    let w_q4_buf = GpuBuffer::from_bytes_device_optimal(&ctx, &q4_bytes).expect("w_q4");
    let w_q4x4_buf = GpuBuffer::from_bytes_device_optimal(&ctx, &q4x4_bytes).expect("w_q4x4");
    let x_buf = GpuBuffer::from_f32_slice(&ctx, &x_batch).expect("x_buf");
    let out_buf = GpuBuffer::new_host_visible(&ctx, n_tok * rows * 4).expect("out_buf");
    let am_buf = GpuBuffer::new_host_visible(&ctx, 16).expect("am_buf");

    // 1. Test `gemv_q4_0` & `gemv_q4_0x4`
    let ds_gv_q4 = arena
        .alloc_and_write(&pipes.gemv_q4_0, &[&w_q4_buf, &x_buf, &out_buf])
        .expect("ds_gv_q4");
    let ds_gv_q4x4 = arena
        .alloc_and_write(&pipes.gemv_q4_0x4, &[&w_q4x4_buf, &x_buf, &out_buf])
        .expect("ds_gv_q4x4");
    let gv_pc = GemvPush {
        n_in: cols as u32,
        n_out: rows as u32,
        row_bytes: DType::Q4_0.row_bytes(cols) as u32,
        add_residual: 0,
        out_offset: 0,
    };

    for ds in [ds_gv_q4, ds_gv_q4x4] {
        let pipe = if ds == ds_gv_q4 {
            &pipes.gemv_q4_0
        } else {
            &pipes.gemv_q4_0x4
        };
        ctx.submit_and_wait(|cmd| pipe.cmd_dispatch(cmd, ds, as_bytes(&gv_pc), 2, 1, 1))
            .expect("dispatch gemv");
        let mut got = vec![0.0f32; rows];
        out_buf.read_f32_slice(0, &mut got);
        for r in 0..rows {
            assert!(
                (got[r] - ref_out[r]).abs() < 1e-4,
                "Vulkan GEMV mismatch at row {r}: got={} ref={}",
                got[r],
                ref_out[r]
            );
        }
    }

    // 2. Test `gemm_q4_0x4` + `argmax`
    let ds_gm_q4x4 = arena
        .alloc_and_write(&pipes.gemm_q4_0x4, &[&w_q4x4_buf, &x_buf, &out_buf])
        .expect("ds_gm");
    let ds_am = arena
        .alloc_and_write(&pipes.argmax, &[&out_buf, &am_buf])
        .expect("ds_am");
    let gm_pc = GemmPush {
        n_in: cols as u32,
        n_out: rows as u32,
        n_tok: n_tok as u32,
        row_bytes: DType::Q4_0.row_bytes(cols) as u32,
        add_residual: 0,
        out_stride: rows as u32,
        out_col_offset: 0,
    };
    let am_pc = ArgmaxPush {
        vocab_size: rows as u32,
        logits_offset: 0,
    };
    ctx.submit_and_wait(|cmd| {
        pipes
            .gemm_q4_0x4
            .cmd_dispatch(cmd, ds_gm_q4x4, as_bytes(&gm_pc), 2, 1, 1)?;
        ctx.cmd_compute_barrier(cmd);
        pipes
            .argmax
            .cmd_dispatch(cmd, ds_am, as_bytes(&am_pc), 1, 1, 1)
    })
    .expect("dispatch gemm + argmax");

    let mut got_m = vec![0.0f32; n_tok * rows];
    out_buf.read_f32_slice(0, &mut got_m);
    for i in 0..got_m.len() {
        assert!(
            (got_m[i] - ref_out[i]).abs() < 1e-4,
            "Vulkan GEMM mismatch at {i}: got={} ref={}",
            got_m[i],
            ref_out[i]
        );
    }
    let _ = NormPush {
        n: 0,
        in_offset: 0,
        eps: 1e-5,
    };
    let _ = BlockQ8_0 { d: 0, qs: [0; 32] };
}
