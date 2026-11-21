//! Compiles `shaders/*.comp` (GLSL 450) into SPIR-V.
//!
//! Lookup order for the compiler: `$TM_GLSLC`, `$VULKAN_SDK/bin/glslc`, `glslc` on `PATH`.
//! When no compiler is available, the committed binaries in `spv/` are used instead, so
//! a plain `cargo build` never requires the Vulkan SDK. Regenerate `spv/` with
//! `crates/tm-vulkan/gen_spv.sh` after editing a shader.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_glslc() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("TM_GLSLC") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        candidates.push(Path::new(&sdk).join("bin").join("glslc"));
    }
    candidates.push(PathBuf::from("glslc"));
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

fn main() {
    println!("cargo:rerun-if-env-changed=TM_GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=spv");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shader_dir = manifest.join("shaders");
    let spv_dir = manifest.join("spv");

    let mut shaders: Vec<PathBuf> = fs::read_dir(&shader_dir)
        .expect("read shaders/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "comp"))
        .collect();
    shaders.sort();

    let glslc = find_glslc();
    for src in &shaders {
        println!("cargo:rerun-if-changed={}", src.display());
        let stem = src.file_stem().expect("stem").to_string_lossy().to_string();
        let dst = out_dir.join(format!("{stem}.spv"));
        match &glslc {
            Some(cc) => {
                let out = Command::new(cc)
                    .args(["--target-env=vulkan1.2", "-O", "-I"])
                    .arg(&shader_dir)
                    .arg("-o")
                    .arg(&dst)
                    .arg(src)
                    .output()
                    .expect("spawn glslc");
                if !out.status.success() {
                    panic!(
                        "glslc failed for {}:\n{}",
                        src.display(),
                        String::from_utf8_lossy(&out.stderr)
                    );
                }
            }
            None => {
                let pre = spv_dir.join(format!("{stem}.spv"));
                fs::copy(&pre, &dst).unwrap_or_else(|e| {
                    panic!(
                        "no glslc found (set TM_GLSLC or VULKAN_SDK) and no prebuilt {}: {e}",
                        pre.display()
                    )
                });
            }
        }
    }
}
