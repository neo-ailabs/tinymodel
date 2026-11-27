use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(explicit) = env::var("NVCC") {
        let p = PathBuf::from(explicit);
        if p.exists() {
            return Some(p);
        }
    }
    for env_key in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(env_key) {
            let p = Path::new(&root).join("bin/nvcc");
            if p.exists() {
                return Some(p);
            }
        }
    }
    for cand in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"] {
        let p = PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }
    if Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("nvcc"));
    }
    None
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let cu_src = manifest_dir.join("kernels/cuda_ops.cu");
    let prebuilt_ptx = manifest_dir.join("ptx/cuda_ops.ptx");
    let dst_ptx = out_dir.join("cuda_ops.ptx");

    println!("cargo:rerun-if-changed={}", cu_src.display());
    println!("cargo:rerun-if-changed={}", prebuilt_ptx.display());
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=TM_CUDA_ARCH");

    let force_prebuilt = env::var("TM_CUDA_USE_PREBUILT_PTX")
        .map(|v| v == "1")
        .unwrap_or(false);

    if !force_prebuilt && let Some(nvcc) = find_nvcc() {
        let arch = env::var("TM_CUDA_ARCH").unwrap_or_else(|_| "compute_75".to_string());
        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                "-O3",
                "--use_fast_math",
                &format!("-arch={arch}"),
                "-o",
            ])
            .arg(&dst_ptx)
            .arg(&cu_src)
            .status()
            .unwrap_or_else(|e| panic!("failed to invoke {}: {e}", nvcc.display()));
        if !status.success() {
            panic!("nvcc failed compiling {}", cu_src.display());
        }
        return;
    }

    if prebuilt_ptx.exists() {
        fs::copy(&prebuilt_ptx, &dst_ptx).expect("failed to copy prebuilt cuda_ops.ptx");
    } else {
        fs::write(
            &dst_ptx,
            ".version 7.0\n.target sm_75\n.address_size 64\n\0",
        )
        .expect("failed to write stub PTX");
    }
}
