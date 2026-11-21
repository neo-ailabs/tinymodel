//! Build-time GLSL 450 -> SPIR-V compiler using pure-Rust `naga` (`ADR-008`).
//! Zero external Vulkan SDK (`glslc` / `shaderc` / CMake) required!

use std::env;
use std::fs;
use std::path::PathBuf;

use naga::back::spv;
use naga::front::glsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{Barrier, Block, Statement};

const SHADERS: &[(&str, &str)] = &[
    ("gemv", include_str!("src/shaders/gemv.comp")),
    ("gemm", include_str!("src/shaders/gemm.comp")),
    ("ops", include_str!("src/shaders/ops.comp")),
];

/// `naga` 24's `glsl-in` maps GLSL `barrier()` to `Statement::Barrier(Barrier::all())` which
/// sets `Barrier::SUB_GROUP`, causing `naga`'s SPIR-V backend to emit `Scope::Subgroup` instead
/// of `Scope::Workgroup`! Strip `Barrier::SUB_GROUP` recursively so every GLSL `barrier()`
/// emits a true SPIR-V 1.0 `OpControlBarrier(Scope::Workgroup, ...)` across all warps.
fn fix_workgroup_barriers(block: &mut Block) {
    for stmt in block.iter_mut() {
        match stmt {
            Statement::Barrier(flags) => {
                *flags = Barrier::WORK_GROUP | Barrier::STORAGE;
            }
            Statement::Block(inner) => fix_workgroup_barriers(inner),
            Statement::If { accept, reject, .. } => {
                fix_workgroup_barriers(accept);
                fix_workgroup_barriers(reject);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    fix_workgroup_barriers(&mut c.body);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                fix_workgroup_barriers(body);
                fix_workgroup_barriers(continuing);
            }
            _ => {}
        }
    }
}

fn compile_glsl_compute(name: &str, common: &str, entry_define: &str, body: &str) -> Vec<u8> {
    let no_gw32 = if (name.starts_with("gemv_") || name.starts_with("gemm_"))
        && matches!(entry_define, "DT_Q4_0" | "DT_Q4_0X4")
        || (name == "gemv_q6_k")
    {
        "#define NO_G_W32 1\n"
    } else {
        ""
    };
    let full_src = format!("#version 450\n#define {entry_define} 1\n{no_gw32}{common}\n{body}\n");
    let mut frontend = glsl::Frontend::default();
    let opts = glsl::Options::from(naga::ShaderStage::Compute);
    let mut module = frontend
        .parse(&opts, &full_src)
        .unwrap_or_else(|errs| panic!("GLSL parse failed for {name} ({entry_define}): {errs:?}"));

    for (_, func) in module.functions.iter_mut() {
        fix_workgroup_barriers(&mut func.body);
    }
    for ep in module.entry_points.iter_mut() {
        fix_workgroup_barriers(&mut ep.function.body);
    }

    // Strict SPIR-V 1.0 capability validation: ONLY `PUSH_CONSTANT` allowed (proves zero Subgroup ops remain!)
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::PUSH_CONSTANT);
    let info = validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("SPIR-V validation failed for {name} ({entry_define}): {e:?}"));

    let spv_opts = spv::Options {
        lang_version: (1, 0),
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &spv_opts, None)
        .unwrap_or_else(|e| panic!("SPIR-V emit failed for {name} ({entry_define}): {e:?}"));

    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

fn main() {
    println!("cargo:rerun-if-changed=src/shaders/common.glsl");
    println!("cargo:rerun-if-changed=src/shaders/gemv.comp");
    println!("cargo:rerun-if-changed=src/shaders/gemm.comp");
    println!("cargo:rerun-if-changed=src/shaders/ops.comp");

    let common = include_str!("src/shaders/common.glsl");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let gemv_variants = [
        ("gemv_q4_0", "DT_Q4_0"),
        ("gemv_q4_0x4", "DT_Q4_0X4"),
        ("gemv_q4_1", "DT_Q4_1"),
        ("gemv_q8_0", "DT_Q8_0"),
        ("gemv_q5_k", "DT_Q5_K"),
        ("gemv_q6_k", "DT_Q6_K"),
        ("gemv_f16", "DT_F16"),
        ("gemv_bf16", "DT_BF16"),
        ("gemv_f32", "DT_F32"),
    ];
    for (out_name, def) in gemv_variants {
        let spv_bytes = compile_glsl_compute(out_name, common, def, SHADERS[0].1);
        fs::write(out_dir.join(format!("{out_name}.spv")), spv_bytes).expect("write gemv spv");
    }

    let gemm_variants = [
        ("gemm_q4_0", "DT_Q4_0"),
        ("gemm_q4_0x4", "DT_Q4_0X4"),
        ("gemm_q4_1", "DT_Q4_1"),
        ("gemm_q8_0", "DT_Q8_0"),
        ("gemm_q5_k", "DT_Q5_K"),
        ("gemm_q6_k", "DT_Q6_K"),
        ("gemm_f16", "DT_F16"),
        ("gemm_bf16", "DT_BF16"),
        ("gemm_f32", "DT_F32"),
    ];
    for (out_name, def) in gemm_variants {
        let spv_bytes = compile_glsl_compute(out_name, common, def, SHADERS[1].1);
        fs::write(out_dir.join(format!("{out_name}.spv")), spv_bytes).expect("write gemm spv");
    }

    let ops_variants = [
        ("embed", "OP_EMBED"),
        ("rms_norm", "OP_RMS_NORM"),
        ("qk_norm", "OP_QK_NORM"),
        ("rope_kv", "OP_ROPE_KV"),
        ("gqa_attn", "OP_GQA_ATTN"),
        ("gqa_attn_batch", "OP_GQA_ATTN_BATCH"),
        ("swiglu", "OP_SWIGLU"),
        ("add_inplace", "OP_ADD_INPLACE"),
        ("argmax", "OP_ARGMAX"),
        ("split_qg_norm", "OP_SPLIT_QG_NORM"),
        ("sigmoid_gate", "OP_SIGMOID_GATE"),
        ("gdn_gates", "OP_GDN_GATES"),
        ("gdn_conv_silu_l2", "OP_GDN_CONV_SILU_L2"),
        ("gdn_recur_gated", "OP_GDN_RECUR_GATED"),
    ];
    for (out_name, def) in ops_variants {
        let spv_bytes = compile_glsl_compute(out_name, common, def, SHADERS[2].1);
        fs::write(out_dir.join(format!("{out_name}.spv")), spv_bytes).expect("write ops spv");
    }
}
