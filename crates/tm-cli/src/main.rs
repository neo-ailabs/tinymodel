#![forbid(unsafe_code)]

//! `tinymodel` CLI: inspect GGUF models, probe hardware/telemetry, run inference,
//! dump multi-token logits, benchmark pp/tg throughput, and profile CPU / Metal GPU execution.

mod probe;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tm_core::probe_process_memory_stats;
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{
    CpuDecodeProfile, LlamaModel, LlamaState, Qwen35DecodeProfile, Qwen35Model, Qwen35State,
    Sampler, SamplerConfig, argmax,
};
use tm_runtime::{CpuModel, Session};
use tm_tokenizer::Tokenizer;

fn print_usage() {
    eprintln!(
        "Usage:
  tinymodel probe [-t <threads>] [--stream-mib <mib>] [--no-stream] [--json]
  tinymodel inspect <model.gguf>
  tinymodel run -m <model.gguf> [-p <prompt> | -f <file>] [-n <tokens>] [-t <threads>] [-c <ctx>] [--metal]
  tinymodel bench -m <model.gguf> [-p <prompt> | -f <file>] [-n <tokens>] [-t <threads>] [-c <ctx>] [--metal] [--json]
  tinymodel profile -m <model.gguf> [-n <tokens>] [-t <threads>] [-c <ctx>] [--metal]
  tinymodel logits -m <model.gguf> -o <out.bin> [-p <prompt> | --tokens 1,2,3] [-t <threads>] [-c <ctx>] [--metal]"
    );
}

fn cmd_inspect(path: PathBuf) -> Result<()> {
    let gguf = GgufFile::open(&path)?;
    println!("File: {}", gguf.path().display());
    println!(
        "GGUF version: {}, alignment: {} bytes",
        gguf.version(),
        gguf.alignment()
    );
    println!("Tensors: {}", gguf.tensor_names().len());
    if let Ok(arch) = gguf.req_str("general.architecture") {
        println!("Architecture: {arch}");
    }
    if let Some(name) = gguf.get_meta("general.name").and_then(|v| v.as_str()) {
        println!("Name: {name}");
    }
    for name in gguf.tensor_names().iter().take(16) {
        if let Some(info) = gguf.tensor_info(name) {
            println!(
                "  {:<32} {:?} {:?} ({} bytes)",
                info.name, info.dims, info.dtype, info.byte_len
            );
        }
    }
    Ok(())
}

struct RunOpts {
    model: PathBuf,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    explicit_tokens: Option<Vec<u32>>,
    out_path: Option<PathBuf>,
    n_predict: usize,
    threads: usize,
    max_ctx: usize,
    prefix_cache_mib: usize,
    use_metal: bool,
    json: bool,
    sampler: SamplerConfig,
}

fn parse_run_opts(args: &[String]) -> Result<RunOpts> {
    let mut model: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_file: Option<PathBuf> = None;
    let mut explicit_tokens: Option<Vec<u32>> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut n_predict = 64usize;
    let mut threads = tm_cpu::recommended_threads();
    let mut max_ctx = 2048usize;
    let mut prefix_cache_mib = 0usize;
    let mut use_metal = false;
    let mut json = false;
    let mut sampler = SamplerConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model = Some(PathBuf::from(
                    args.get(i).context("missing value for -m/--model")?,
                ));
            }
            "-p" | "--prompt" => {
                i += 1;
                prompt = Some(
                    args.get(i)
                        .context("missing value for -p/--prompt")?
                        .clone(),
                );
            }
            "-f" | "--file" => {
                i += 1;
                prompt_file = Some(PathBuf::from(
                    args.get(i).context("missing value for -f/--file")?,
                ));
            }
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -o/--output")?,
                ));
            }
            "--tokens" => {
                i += 1;
                let raw = args.get(i).context("missing value for --tokens")?;
                let mut ids = Vec::new();
                for part in raw.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        ids.push(
                            trimmed
                                .parse::<u32>()
                                .context("invalid token id in --tokens")?,
                        );
                    }
                }
                explicit_tokens = Some(ids);
            }
            "-n" | "--n-predict" => {
                i += 1;
                n_predict = args
                    .get(i)
                    .context("missing value for -n")?
                    .parse()
                    .context("invalid -n")?;
            }
            "-t" | "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .context("missing value for -t")?
                    .parse()
                    .context("invalid -t")?;
            }
            "-c" | "--ctx-size" => {
                i += 1;
                max_ctx = args
                    .get(i)
                    .context("missing value for -c")?
                    .parse()
                    .context("invalid -c")?;
            }
            "--temp" => {
                i += 1;
                sampler.temperature = args
                    .get(i)
                    .context("missing value for --temp")?
                    .parse()
                    .context("invalid --temp")?;
            }
            "--top-k" => {
                i += 1;
                sampler.top_k = args
                    .get(i)
                    .context("missing value for --top-k")?
                    .parse()
                    .context("invalid --top-k")?;
            }
            "--top-p" => {
                i += 1;
                sampler.top_p = args
                    .get(i)
                    .context("missing value for --top-p")?
                    .parse()
                    .context("invalid --top-p")?;
            }
            "--repeat-penalty" => {
                i += 1;
                sampler.repeat_penalty = args
                    .get(i)
                    .context("missing value for --repeat-penalty")?
                    .parse()
                    .context("invalid --repeat-penalty")?;
            }
            "-s" | "--seed" => {
                i += 1;
                sampler.seed = args
                    .get(i)
                    .context("missing value for --seed")?
                    .parse()
                    .context("invalid --seed")?;
            }
            "--metal" => {
                use_metal = true;
            }
            "--prefix-cache-mib" => {
                i += 1;
                prefix_cache_mib = args
                    .get(i)
                    .context("missing value for --prefix-cache-mib")?
                    .parse()
                    .context("invalid --prefix-cache-mib")?;
            }
            "--json" => {
                json = true;
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    let model = model.context("missing required -m <model.gguf>")?;
    Ok(RunOpts {
        model,
        prompt,
        prompt_file,
        explicit_tokens,
        out_path,
        n_predict,
        threads,
        max_ctx,
        prefix_cache_mib,
        use_metal,
        json,
        sampler,
    })
}

fn resolve_prompt_ids(opts: &RunOpts, tok: &Tokenizer) -> Result<Vec<u32>> {
    if let Some(ref ids) = opts.explicit_tokens {
        if ids.is_empty() {
            bail!("--tokens must contain at least 1 token id");
        }
        return Ok(ids.clone());
    }
    let prompt_text = if let Some(ref fpath) = opts.prompt_file {
        std::fs::read_to_string(fpath)
            .with_context(|| format!("failed to read prompt file '{}'", fpath.display()))?
    } else {
        opts.prompt.clone().unwrap_or_else(|| "Hello".to_string())
    };
    let ids = tok.encode(&prompt_text, tok.add_bos);
    if ids.is_empty() {
        bail!("prompt encoded to 0 tokens");
    }
    Ok(ids)
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn print_metal_profile_report(
    model_name: &str,
    dev_name: &str,
    rep: &tm_metal::MetalProfileReport,
) {
    let dec_tps = 1e6 / rep.decode_single_cb_wall_us.max(1e-3);
    let host_ovh_us = (rep.decode_single_cb_wall_us - rep.decode_single_cb_gpu_us).max(0.0);
    let sum_dec_gpu_us: f64 = rep.decode_stages.iter().map(|s| s.gpu_us).sum();
    let total_dec_bytes: u64 = rep.decode_stages.iter().map(|s| s.bytes).sum();
    let e2e_bw_gbs = (total_dec_bytes as f64 / 1e9) / (rep.decode_single_cb_wall_us * 1e-6);

    println!("=== Metal Profile: {model_name} ({dev_name}) ===");
    println!(
        "Decode 1-CB E2E: wall={:.1} us/tok ({:.2} tok/s) | gpu={:.1} us/tok | host+sync={:.1} us/tok ({:.1}%) | eff_bw={:.2} GB/s",
        rep.decode_single_cb_wall_us,
        dec_tps,
        rep.decode_single_cb_gpu_us,
        host_ovh_us,
        100.0 * host_ovh_us / rep.decode_single_cb_wall_us.max(1e-3),
        e2e_bw_gbs,
    );
    println!();
    println!(
        "{:<28} {:>10} {:>8} {:>10} {:>12} {:>14}",
        "Decode Stage", "GPU us", "Share%", "MiB/tok", "Eff GB/s", "Eff GFLOP/s"
    );
    println!("{}", "-".repeat(86));
    for st in &rep.decode_stages {
        let pct = 100.0 * st.gpu_us / sum_dec_gpu_us.max(1e-3);
        let mib = st.bytes as f64 / (1024.0 * 1024.0);
        let gbs = if st.bytes > 0 {
            (st.bytes as f64 / 1e9) / (st.gpu_us * 1e-6)
        } else {
            0.0
        };
        let gflops = if st.flops > 0 {
            (st.flops as f64 / 1e9) / (st.gpu_us * 1e-6)
        } else {
            0.0
        };
        println!(
            "{:<28} {:>10.1} {:>7.1}% {:>10.2} {:>12.2} {:>14.1}",
            st.name, st.gpu_us, pct, mib, gbs, gflops
        );
    }
    println!(
        "{:<28} {:>10.1} {:>7.1}% {:>10.2}",
        "Sum (multi-CB)",
        sum_dec_gpu_us,
        100.0,
        total_dec_bytes as f64 / (1024.0 * 1024.0)
    );

    let pp_tps = (rep.prefill_tokens as f64) / (rep.prefill_wall_ms * 1e-3).max(1e-9);
    println!();
    println!(
        "Prefill pp{}: wall={:.2} ms ({:.1} tok/s)",
        rep.prefill_tokens, rep.prefill_wall_ms, pp_tps
    );
    if !rep.prefill_chunk_stages.is_empty() {
        let sum_pp_gpu_us: f64 = rep.prefill_chunk_stages.iter().map(|s| s.gpu_us).sum();
        println!(
            "{:<28} {:>10} {:>8} {:>10} {:>12} {:>14}",
            "Prefill Stage", "GPU ms", "Share%", "MiB", "Eff GB/s", "Eff GFLOP/s"
        );
        println!("{}", "-".repeat(86));
        for st in &rep.prefill_chunk_stages {
            let pct = 100.0 * st.gpu_us / sum_pp_gpu_us.max(1e-3);
            let mib = st.bytes as f64 / (1024.0 * 1024.0);
            let gbs = if st.bytes > 0 {
                (st.bytes as f64 / 1e9) / (st.gpu_us * 1e-6)
            } else {
                0.0
            };
            let gflops = if st.flops > 0 {
                (st.flops as f64 / 1e9) / (st.gpu_us * 1e-6)
            } else {
                0.0
            };
            println!(
                "{:<28} {:>10.3} {:>7.1}% {:>10.2} {:>12.2} {:>14.1}",
                st.name,
                st.gpu_us * 1e-3,
                pct,
                mib,
                gbs,
                gflops
            );
        }
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn cmd_run_metal(opts: RunOpts) -> Result<()> {
    let t_load = Instant::now();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let tok = Tokenizer::from_gguf(&gguf)?;
    let prompt_ids = resolve_prompt_ids(&opts, &tok)?;
    let is_pure_greedy =
        opts.sampler.temperature <= 0.0 && (opts.sampler.repeat_penalty - 1.0).abs() < 1e-6;

    let mut runner = tm_metal::MetalRunner::from_gguf(gguf, opts.max_ctx)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[tinymodel] arch={} backend=metal ({}) weights={} load={:.1}ms prompt_tokens={} vocab={} layers={}",
        runner.arch(),
        runner.device_name(),
        if runner.is_zero_copy_weights() {
            "zero-copy"
        } else {
            "copied"
        },
        load_ms,
        prompt_ids.len(),
        runner.vocab_size(),
        runner.n_layers()
    );

    let mut sampler = Sampler::new(opts.sampler, runner.vocab_size());
    let mut history = prompt_ids.clone();

    let t_pp = Instant::now();
    let logits = runner.forward_batch(&prompt_ids, 0, true)?;
    let mut next_id = sampler.sample(logits, &history);
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
    let mut pos = prompt_ids.len();
    let mut generated = 0usize;

    let t_gen = Instant::now();
    while generated < opts.n_predict {
        if tok.is_stop_token(next_id) {
            break;
        }
        stdout.write_all(&tok.decode_token_bytes(next_id))?;
        stdout.flush()?;
        history.push(next_id);
        generated += 1;

        if generated >= opts.n_predict {
            break;
        }
        if is_pure_greedy {
            next_id = runner.forward_token_greedy(next_id, pos)?;
        } else {
            let logits = runner.forward_token(next_id, pos, true)?;
            next_id = sampler.sample(logits, &history);
        }
        pos += 1;
    }
    let gen_sec = t_gen.elapsed().as_secs_f64();

    writeln!(stdout)?;
    let pp_tps = (prompt_ids.len() as f64) / pp_sec.max(1e-9);
    let tg_runs = generated.saturating_sub(1);
    let tg_tps = if tg_runs > 0 {
        (tg_runs as f64) / gen_sec.max(1e-9)
    } else {
        0.0
    };
    eprintln!(
        "[tinymodel] prompt: {:.2} ms / {} tok ({:.2} tok/s) | eval: {:.2} ms / {} runs ({:.2} tok/s)",
        pp_sec * 1e3,
        prompt_ids.len(),
        pp_tps,
        gen_sec * 1e3,
        tg_runs,
        tg_tps
    );
    Ok(())
}

fn compute_active_decode_bytes(gguf: &GgufFile) -> u64 {
    let mut total = 0u64;
    let has_separate_output = gguf.tensor_info("output.weight").is_some();
    for name in gguf.tensor_names() {
        if let Some(info) = gguf.tensor_info(name) {
            if has_separate_output && name == "token_embd.weight" {
                // In decode, untied `token_embd.weight` only reads 1 row; exclude from streaming weight bytes.
                if let Some(&cols) = info.dims.first() {
                    total += info.dtype.row_bytes(cols) as u64;
                }
                continue;
            }
            if name == "per_layer_token_embd.weight" {
                // Gemma 4 PLE lookup table only reads 1 row per token (`ARCHITECTURE.md` §1)!
                if let Some(&cols) = info.dims.first() {
                    total += info.dtype.row_bytes(cols) as u64;
                }
                continue;
            }
            total += info.byte_len as u64;
        }
    }
    total
}

fn resolve_bench_pp_ids(opts: &RunOpts, gguf: &GgufFile) -> Result<(Vec<u32>, bool)> {
    let has_custom =
        opts.prompt.is_some() || opts.prompt_file.is_some() || opts.explicit_tokens.is_some();
    if has_custom {
        let tok = Tokenizer::from_gguf(gguf)?;
        let ids = resolve_prompt_ids(opts, &tok)?;
        if ids.len() + opts.n_predict + 2 > opts.max_ctx {
            bail!(
                "prompt ({} tokens) + n_predict ({}) exceeds max_ctx ({})",
                ids.len(),
                opts.n_predict,
                opts.max_ctx
            );
        }
        Ok((ids, true))
    } else {
        let n_pp = 512usize.min(opts.max_ctx.saturating_sub(opts.n_predict + 2));
        let ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
        Ok((ids, false))
    }
}

struct BenchMetrics<'a> {
    model_name: &'a str,
    arch: &'a str,
    backend: &'a str,
    isa: &'a str,
    device_name: &'a str,
    weights_mode: &'a str,
    threads: usize,
    max_ctx: usize,
    corpus_mode: bool,
    active_decode_bytes: u64,
    load_ms: f64,
    ttft_ms: f64,
    n_pp: usize,
    pp_ms: f64,
    n_gen: usize,
    tg_ms: f64,
    last_tok: u32,
    greedy_hash: String,
    tel_before: probe::TelemetrySnapshot,
    tel_after: probe::TelemetrySnapshot,
}

impl<'a> BenchMetrics<'a> {
    fn print_json(&self) {
        let pp_tps = (self.n_pp as f64) / (self.pp_ms * 1e-3).max(1e-9);
        let tg_tps = (self.n_gen as f64) / (self.tg_ms * 1e-3).max(1e-9);
        let ms_per_tok = if self.n_gen > 0 {
            self.tg_ms / (self.n_gen as f64)
        } else {
            0.0
        };
        let eff_bw_gbs = if self.n_gen > 0 && self.tg_ms > 0.0 {
            ((self.active_decode_bytes as f64) * (self.n_gen as f64) / (self.tg_ms * 1e-3)) / 1e9
        } else {
            0.0
        };
        let mstats = probe_process_memory_stats();
        println!(
            "{{\"schema_version\":1,\"model\":\"{}\",\"arch\":\"{}\",\"backend\":\"{}\",\
            \"isa\":\"{}\",\"device_name\":\"{}\",\"weights_mode\":\"{}\",\
            \"threads\":{},\"max_ctx\":{},\"corpus_mode\":{},\"active_decode_bytes\":{},\
            \"load_ms\":{:.2},\"ttft_ms\":{:.2},\"pp_tokens\":{},\"pp_ms\":{:.2},\"pp_tps\":{:.2},\
            \"tg_tokens\":{},\"tg_ms\":{:.2},\"tg_tps\":{:.2},\"ms_per_tok\":{:.3},\
            \"decode_eff_bw_gbs\":{:.2},\"last_tok\":{},\"greedy_tokens_fnv1a64\":\"{}\",\
            \"rss_bytes\":{},\"peak_rss_bytes\":{},\"private_dirty_bytes\":{},\"pss_anon_bytes\":{},\
            \"telemetry_before\":{},\"telemetry_after\":{}}}",
            probe::json_escape(self.model_name),
            probe::json_escape(self.arch),
            probe::json_escape(self.backend),
            probe::json_escape(self.isa),
            probe::json_escape(self.device_name),
            probe::json_escape(self.weights_mode),
            self.threads,
            self.max_ctx,
            self.corpus_mode,
            self.active_decode_bytes,
            self.load_ms,
            self.ttft_ms,
            self.n_pp,
            self.pp_ms,
            pp_tps,
            self.n_gen,
            self.tg_ms,
            tg_tps,
            ms_per_tok,
            eff_bw_gbs,
            self.last_tok,
            self.greedy_hash,
            probe::opt_u64_json(mstats.rss_bytes),
            probe::opt_u64_json(mstats.peak_rss_bytes),
            probe::opt_u64_json(mstats.private_dirty_bytes),
            probe::opt_u64_json(mstats.pss_anon_bytes),
            self.tel_before.to_json(),
            self.tel_after.to_json(),
        );
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn cmd_bench_metal(opts: RunOpts) -> Result<()> {
    let tel_before = probe::collect_telemetry();
    let t_load = Instant::now();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let arch = gguf.req_str("general.architecture")?.to_string();
    let active_decode_bytes = compute_active_decode_bytes(&gguf);
    let (pp_ids, corpus_mode) = resolve_bench_pp_ids(&opts, &gguf)?;
    let n_pp = pp_ids.len();
    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    let mut runner = tm_metal::MetalRunner::from_gguf(gguf, opts.max_ctx)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let _ = runner.forward_token_greedy(1, 0)?;
    let _ = runner.forward_token_greedy(2, 1)?;
    runner.reset_state();
    let warmup_ids = [1u32; 8];
    let _ = runner.forward_batch(&warmup_ids, 0, true)?;
    runner.reset_state();

    let t_pp = Instant::now();
    let pp_logits = runner.forward_batch(&pp_ids, 0, corpus_mode)?;
    let first_tok = if corpus_mode { argmax(pp_logits) } else { 0 };
    let pp_sec = t_pp.elapsed().as_secs_f64();
    let pp_tps = (n_pp as f64) / pp_sec.max(1e-9);

    let n_gen = opts.n_predict;
    let mut gen_tokens = Vec::with_capacity(n_gen + 1);
    let (mut tok, base_pos) = if corpus_mode {
        (first_tok, n_pp)
    } else {
        runner.reset_state();
        (runner.forward_token_greedy(1, 0)?, 1)
    };
    gen_tokens.push(tok);
    let t0 = Instant::now();
    for step in 0..n_gen {
        tok = runner.forward_token_greedy(tok, base_pos + step)?;
        gen_tokens.push(tok);
    }
    let sec = t0.elapsed().as_secs_f64();
    let tps = (n_gen as f64) / sec.max(1e-9);
    let tel_after = probe::collect_telemetry();
    let dev_name = runner.device_name();
    let wmode = if runner.is_zero_copy_weights() {
        "zero-copy"
    } else {
        "copied"
    };
    if opts.json {
        BenchMetrics {
            model_name: &model_name,
            arch: &arch,
            backend: "metal",
            isa: "metal",
            device_name: &dev_name,
            weights_mode: wmode,
            threads: 1,
            max_ctx: opts.max_ctx,
            corpus_mode,
            active_decode_bytes,
            load_ms,
            ttft_ms: pp_sec * 1e3,
            n_pp,
            pp_ms: pp_sec * 1e3,
            n_gen,
            tg_ms: sec * 1e3,
            last_tok: tok,
            greedy_hash: probe::hash_token_sequence(&gen_tokens),
            tel_before,
            tel_after,
        }
        .print_json();
    } else {
        println!(
            "bench: model={} arch={} backend=metal ({}) weights={} load={:.1}ms pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
            model_name,
            arch,
            dev_name,
            wmode,
            load_ms,
            n_pp,
            pp_tps,
            n_gen,
            tps,
            (sec * 1e3) / (n_gen as f64),
            tok
        );
    }
    Ok(())
}

/// On Linux (including Raspberry Pi & Docker cgroup v2), verifies that runtime dirty-page
/// buffers (KV cache + GDN recurrent state + scratch) fit comfortably inside `available_bytes`
/// before allocating, preventing SD-swap thrashing or container OOM kills.
fn check_linux_dirty_budget(required_bytes: u64, max_ctx: usize) -> Result<()> {
    const OS_HEADROOM_BYTES: u64 = 32 * 1024 * 1024;
    if let Ok(budget) = tm_core::probe_system_memory() {
        let safe_limit = budget.available_bytes.saturating_sub(OS_HEADROOM_BYTES);
        if required_bytes > safe_limit {
            let cg_note = budget
                .cgroup_max_bytes
                .map(|m| format!(" (cgroup memory.max={:.1} MiB)", m as f64 / 1048576.0))
                .unwrap_or_default();
            bail!(
                "insufficient memory for runtime state (-c {max_ctx}): need {:.1} MiB dirty RAM, \
                 only {:.1} MiB safely available{}; reduce -c <ctx> or free system memory",
                required_bytes as f64 / 1048576.0,
                safe_limit as f64 / 1048576.0,
                cg_note,
            );
        }
    }
    Ok(())
}

fn estimate_llama_dirty_bytes(cfg: &tm_models::LlamaConfig, max_ctx: usize) -> u64 {
    let kv_bytes =
        (2 * cfg.n_layers * max_ctx * cfg.n_head_kv * cfg.head_dim * core::mem::size_of::<u16>())
            as u64;
    let scratch_bytes = ((cfg.vocab_size + 36 * (cfg.n_embd + cfg.n_ff)) * 4) as u64;
    kv_bytes + scratch_bytes
}

fn cmd_run(opts: RunOpts) -> Result<()> {
    if opts.use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        return cmd_run_metal(opts);
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
    }
    let t_load = Instant::now();
    let gguf = GgufFile::open(&opts.model)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let prompt_ids = resolve_prompt_ids(&opts, &tok)?;
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();

    let model = CpuModel::from_gguf(&gguf)?;
    check_linux_dirty_budget(model.estimate_dirty_bytes(opts.max_ctx), opts.max_ctx)?;
    let cache_bytes = opts.prefix_cache_mib * 1024 * 1024;
    let mut session = Session::new(&model, opts.max_ctx, cache_bytes);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "[tinymodel] arch={} isa={} threads={} load={:.1}ms prompt_tokens={} vocab={} layers={} prefix_cache_mib={}",
        model.arch(),
        kt.isa.name(),
        pool.num_threads(),
        load_ms,
        prompt_ids.len(),
        model.vocab_size(),
        model.n_layers(),
        opts.prefix_cache_mib
    );

    let t_pp = Instant::now();
    if cache_bytes > 0 {
        let rep = session.prefill_with_cache(&pool, &prompt_ids, Some(64))?;
        if rep.cached_tokens > 0 {
            eprintln!(
                "[tinymodel] prefix_cache hit: reused {}/{} tokens",
                rep.cached_tokens, rep.total_tokens
            );
        }
    } else {
        model.forward_batch(&mut session.state, &pool, kt, &prompt_ids, 0, true)?;
    }
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
    let mut sampler = Sampler::new(opts.sampler, model.vocab_size());
    let mut history = prompt_ids.clone();
    let mut next_id = sampler.sample(session.logits(), &history);
    let mut pos = prompt_ids.len();
    let mut generated = 0usize;

    let t_gen = Instant::now();
    while generated < opts.n_predict {
        if tok.is_stop_token(next_id) {
            break;
        }
        stdout.write_all(&tok.decode_token_bytes(next_id))?;
        stdout.flush()?;
        history.push(next_id);
        generated += 1;

        if generated >= opts.n_predict {
            break;
        }
        model.forward_token(&mut session.state, &pool, kt, next_id, pos, true)?;
        next_id = sampler.sample(session.logits(), &history);
        pos += 1;
    }
    let gen_sec = t_gen.elapsed().as_secs_f64();

    writeln!(stdout)?;
    let pp_tps = (prompt_ids.len() as f64) / pp_sec.max(1e-9);
    let tg_runs = generated.saturating_sub(1);
    let tg_tps = if tg_runs > 0 {
        (tg_runs as f64) / gen_sec.max(1e-9)
    } else {
        0.0
    };
    eprintln!(
        "[tinymodel] prompt: {:.2} ms / {} tok ({:.2} tok/s) | eval: {:.2} ms / {} runs ({:.2} tok/s)",
        pp_sec * 1e3,
        prompt_ids.len(),
        pp_tps,
        gen_sec * 1e3,
        tg_runs,
        tg_tps
    );

    Ok(())
}

fn cmd_bench(opts: RunOpts) -> Result<()> {
    if opts.use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        return cmd_bench_metal(opts);
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
    }
    let tel_before = probe::collect_telemetry();
    let t_load = Instant::now();
    let gguf = GgufFile::open(&opts.model)?;
    let arch = gguf.req_str("general.architecture")?.to_string();
    let active_decode_bytes = compute_active_decode_bytes(&gguf);
    let (pp_ids, corpus_mode) = resolve_bench_pp_ids(&opts, &gguf)?;
    let n_pp = pp_ids.len();
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();
    let dev_model_name = probe::detect_device_model_name();
    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    let model = CpuModel::from_gguf(&gguf)?;
    check_linux_dirty_budget(model.estimate_dirty_bytes(opts.max_ctx), opts.max_ctx)?;
    let mut state = model.new_state(opts.max_ctx);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    // Warm up mmap pages & thread pool
    model.forward_token(&mut state, &pool, kt, 1, 0, true)?;
    state.reset();
    model.forward_batch(&mut state, &pool, kt, &[1u32; 8], 0, false)?;
    state.reset();

    let t_pp = Instant::now();
    model.forward_batch(&mut state, &pool, kt, &pp_ids, 0, corpus_mode)?;
    let first_tok = if corpus_mode {
        argmax(state.logits())
    } else {
        0
    };
    let pp_sec = t_pp.elapsed().as_secs_f64();
    let pp_tps = (n_pp as f64) / pp_sec.max(1e-9);

    let n_gen = opts.n_predict;
    let mut gen_tokens = Vec::with_capacity(n_gen + 1);
    let (mut tok, base_pos) = if corpus_mode {
        (first_tok, n_pp)
    } else {
        state.reset();
        model.forward_token(&mut state, &pool, kt, 1, 0, true)?;
        (argmax(state.logits()), 1)
    };
    gen_tokens.push(tok);

    let t0 = Instant::now();
    for step in 0..n_gen {
        model.forward_token(&mut state, &pool, kt, tok, base_pos + step, true)?;
        tok = argmax(state.logits());
        gen_tokens.push(tok);
    }
    let sec = t0.elapsed().as_secs_f64();
    let tps = (n_gen as f64) / sec.max(1e-9);
    let tel_after = probe::collect_telemetry();
    if opts.json {
        BenchMetrics {
            model_name: &model_name,
            arch: &arch,
            backend: "cpu",
            isa: kt.isa.name(),
            device_name: &dev_model_name,
            weights_mode: "mmap",
            threads: pool.num_threads(),
            max_ctx: opts.max_ctx,
            corpus_mode,
            active_decode_bytes,
            load_ms,
            ttft_ms: pp_sec * 1e3,
            n_pp,
            pp_ms: pp_sec * 1e3,
            n_gen,
            tg_ms: sec * 1e3,
            last_tok: tok,
            greedy_hash: probe::hash_token_sequence(&gen_tokens),
            tel_before,
            tel_after,
        }
        .print_json();
    } else {
        println!(
            "bench: model={} arch={} isa={} threads={} pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
            model_name,
            arch,
            kt.isa.name(),
            pool.num_threads(),
            n_pp,
            pp_tps,
            n_gen,
            tps,
            (sec * 1e3) / (n_gen as f64),
            tok
        );
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn cmd_profile_metal(opts: RunOpts) -> Result<()> {
    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let n_runs = opts.n_predict.max(16);
    let mut runner = tm_metal::MetalRunner::from_gguf(gguf, opts.max_ctx)?;
    let dev = runner.device_name();
    let rep = runner.profile(n_runs, 512)?;
    print_metal_profile_report(&model_name, &dev, &rep);
    Ok(())
}

fn cmd_profile(opts: RunOpts) -> Result<()> {
    if opts.use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        return cmd_profile_metal(opts);
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
    }

    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?");
    let gguf = GgufFile::open(&opts.model)?;
    let arch = gguf.req_str("general.architecture")?.to_string();
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();

    if arch == "qwen35" {
        let model = Qwen35Model::from_gguf(&gguf)?;
        let mut state = Qwen35State::new(&model, opts.max_ctx);
        for i in 0..4 {
            model.forward_token(&mut state, &pool, kt, 1, i, true)?;
        }
        state.reset();
        let n_pp = 256usize.min(opts.max_ctx / 2);
        let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
        let mut pp_prof = Qwen35DecodeProfile::default();
        let t_pp = Instant::now();
        model.forward_batch_profiled(&mut state, &pool, kt, &pp_ids, 0, true, &mut pp_prof)?;
        let pp_total_ns = t_pp.elapsed().as_nanos() as u64;
        let pp_ms = pp_total_ns as f64 / 1e6;
        let pp_tps = (n_pp as f64) / (pp_ms * 1e-3).max(1e-9);

        let n_runs = opts.n_predict.max(16);
        let mut prof = Qwen35DecodeProfile::default();
        let mut tok = argmax(state.logits());
        let t_dec = Instant::now();
        for step in 0..n_runs {
            model.forward_token_profiled(&mut state, &pool, kt, tok, n_pp + step, &mut prof)?;
            tok = argmax(state.logits());
        }
        let total_ns = t_dec.elapsed().as_nanos() as u64;
        let runs_f = prof.n_steps as f64;
        let total_us = (total_ns as f64 / 1e3) / runs_f;
        let dec_tps = 1e6 / total_us.max(1e-3);

        println!(
            "=== CPU Profile (Qwen3.5 Hybrid): {} (isa={}, threads={}) ===",
            model_name,
            kt.isa.name(),
            pool.num_threads()
        );
        println!(
            "Prefill pp{}: {:.2} ms ({:.2} tok/s) | Decode tg{}: {:.1} us/tok ({:.2} tok/s)",
            n_pp, pp_ms, pp_tps, n_runs, total_us, dec_tps
        );
        println!("{:<22} {:>10} {:>8}", "Prefill Stage", "Total ms", "Share%");
        println!("{}", "-".repeat(44));
        let pp_rows = [
            ("embed+rope_tbl", pp_prof.embed_ns),
            ("rms_norm", pp_prof.rms_norm_ns),
            ("gdn_proj", pp_prof.gdn_proj_ns),
            ("gdn_conv_l2", pp_prof.gdn_conv_norm_ns),
            ("gdn_recur+gnorm", pp_prof.gdn_recur_ns),
            ("gdn_ssm_out (Q5_K)", pp_prof.gdn_out_ns),
            ("full_qkv", pp_prof.full_qkv_ns),
            ("full_attn", pp_prof.full_attn_ns),
            ("full_wo", pp_prof.full_wo_ns),
            ("ffn_gate_up+silu", pp_prof.ffn_gate_up_ns),
            ("ffn_down", pp_prof.ffn_down_ns),
            ("lm_head (Q6_K)", pp_prof.lm_head_ns),
        ];
        for (name, ns) in pp_rows {
            let ms = ns as f64 / 1e6;
            let pct = 100.0 * (ns as f64) / (pp_total_ns as f64).max(1.0);
            println!("{:<22} {:>10.2} {:>7.1}%", name, ms, pct);
        }
        println!();
        println!("{:<22} {:>10} {:>8}", "Decode Stage", "Wall us", "Share%");
        println!("{}", "-".repeat(44));
        let rows = [
            ("embed+rope_tbl", prof.embed_ns),
            ("rms_norm", prof.rms_norm_ns),
            ("gdn_proj", prof.gdn_proj_ns),
            ("gdn_conv_l2", prof.gdn_conv_norm_ns),
            ("gdn_recur+gnorm", prof.gdn_recur_ns),
            ("gdn_ssm_out (Q5_K)", prof.gdn_out_ns),
            ("full_qkv", prof.full_qkv_ns),
            ("full_attn", prof.full_attn_ns),
            ("full_wo", prof.full_wo_ns),
            ("ffn_gate_up+silu", prof.ffn_gate_up_ns),
            ("ffn_down", prof.ffn_down_ns),
            ("lm_head (Q6_K)", prof.lm_head_ns),
        ];
        for (name, ns) in rows {
            let us = (ns as f64 / 1e3) / runs_f;
            let pct = 100.0 * (ns as f64) / (total_ns as f64).max(1.0);
            println!("{:<22} {:>10.1} {:>7.1}%", name, us, pct);
        }
        return Ok(());
    }

    let model = LlamaModel::from_gguf(&gguf)?;
    check_linux_dirty_budget(
        estimate_llama_dirty_bytes(&model.cfg, opts.max_ctx),
        opts.max_ctx,
    )?;
    let mut state = LlamaState::new(&model.cfg, opts.max_ctx);

    for i in 0..4 {
        model.forward_token(&mut state, &pool, kt, 1, i, true)?;
    }

    let n_pp = 256usize.min(opts.max_ctx / 2);
    let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
    let mut pp_prof = CpuDecodeProfile::default();
    let t_pp = Instant::now();
    model.forward_batch_profiled(&mut state, &pool, kt, &pp_ids, 0, true, &mut pp_prof)?;
    let pp_ms = t_pp.elapsed().as_secs_f64() * 1e3;
    let pp_tps = (n_pp as f64) / (pp_ms * 1e-3).max(1e-9);

    let n_runs = opts.n_predict.max(16);
    let mut prof = CpuDecodeProfile::default();
    let mut tok = argmax(state.logits());
    for step in 0..n_runs {
        model.forward_token_profiled(&mut state, &pool, kt, tok, n_pp + step, true, &mut prof)?;
        tok = argmax(state.logits());
    }

    let runs_f = prof.runs as f64;
    let total_us = (prof.total_ns as f64 / 1e3) / runs_f;
    let dec_tps = 1e6 / total_us.max(1e-3);
    let total_bytes = (prof.embed_bytes
        + prof.attn_qkv_bytes
        + prof.gqa_kv_bytes
        + prof.attn_wo_bytes
        + prof.ffn_gate_up_bytes
        + prof.ffn_down_bytes
        + prof.lm_head_bytes) as f64
        / runs_f;
    let e2e_gbs = (total_bytes / 1e9) / (total_us * 1e-6);

    println!(
        "=== CPU Profile: {} (isa={}, threads={}, lm_head={:?}) ===",
        model_name,
        kt.isa.name(),
        pool.num_threads(),
        model.lm_head_dtype()
    );
    println!(
        "Prefill pp{}: {:.2} ms ({:.2} tok/s) | Decode tg{}: {:.1} us/tok ({:.2} tok/s, eff_bw={:.2} GB/s)",
        n_pp, pp_ms, pp_tps, n_runs, total_us, dec_tps, e2e_gbs
    );
    println!();
    println!("{:<20} {:>10} {:>8}", "Prefill Stage", "Total ms", "Share%");
    println!("{}", "-".repeat(42));
    let pp_rows: [(&str, u64); 11] = [
        ("embed+rope_tbl", pp_prof.embed_ns),
        ("attn_norm", pp_prof.attn_norm_ns),
        ("attn_qkv", pp_prof.attn_qkv_ns),
        ("rope+kv_store", pp_prof.rope_kv_ns),
        ("gqa_attn", pp_prof.gqa_attn_ns),
        ("attn_wo+res", pp_prof.attn_wo_ns),
        ("ffn_norm", pp_prof.ffn_norm_ns),
        ("ffn_gate_up", pp_prof.ffn_gate_up_ns),
        ("swiglu", pp_prof.swiglu_ns),
        ("ffn_down+res", pp_prof.ffn_down_ns),
        ("lm_head", pp_prof.lm_head_ns),
    ];
    for (name, ns) in pp_rows {
        if ns == 0 {
            continue;
        }
        let ms = ns as f64 / 1e6;
        let pct = 100.0 * (ns as f64) / (pp_prof.total_ns as f64).max(1.0);
        println!("{:<20} {:>10.2} {:>7.1}%", name, ms, pct);
    }
    println!();
    println!(
        "{:<20} {:>10} {:>8} {:>10} {:>12}",
        "Decode Stage", "Wall us", "Share%", "MiB/tok", "Eff GB/s"
    );
    println!("{}", "-".repeat(64));

    let rows: [(&str, u64, u64); 13] = [
        ("embed+rope_tbl", prof.embed_ns, prof.embed_bytes),
        ("attn_norm", prof.attn_norm_ns, 0),
        ("attn_qkv", prof.attn_qkv_ns, prof.attn_qkv_bytes),
        ("qk_norm", prof.qk_norm_ns, 0),
        ("rope+kv_store", prof.rope_kv_ns, 0),
        ("gqa_attn", prof.gqa_attn_ns, prof.gqa_kv_bytes),
        ("attn_wo+res", prof.attn_wo_ns, prof.attn_wo_bytes),
        ("ffn_norm", prof.ffn_norm_ns, 0),
        ("ffn_gate_up", prof.ffn_gate_up_ns, prof.ffn_gate_up_bytes),
        ("swiglu", prof.swiglu_ns, 0),
        ("ffn_down+res", prof.ffn_down_ns, prof.ffn_down_bytes),
        ("output_norm", prof.output_norm_ns, 0),
        ("lm_head", prof.lm_head_ns, prof.lm_head_bytes),
    ];

    for (name, ns, bytes) in rows {
        if ns == 0 {
            continue;
        }
        let us = (ns as f64 / 1e3) / runs_f;
        let pct = 100.0 * (ns as f64) / (prof.total_ns as f64).max(1.0);
        let b_per_tok = bytes as f64 / runs_f;
        let mib = b_per_tok / (1024.0 * 1024.0);
        let gbs = if b_per_tok > 0.0 {
            (b_per_tok / 1e9) / (us * 1e-6)
        } else {
            0.0
        };
        println!(
            "{:<20} {:>10.1} {:>7.1}% {:>10.2} {:>12.2}",
            name, us, pct, mib, gbs
        );
    }
    Ok(())
}

fn write_f32_le(w: &mut BufWriter<File>, slice: &[f32]) -> Result<()> {
    for &v in slice {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Evaluates a multi-token sequence and writes full `[n_steps, vocab_size]` FP32 logits
/// in `TMLG` v1 binary format for `tools/logits_diff.py` (`AGENTS.md` Rule #7).
fn cmd_logits(opts: RunOpts) -> Result<()> {
    let out_path = opts
        .out_path
        .clone()
        .context("logits requires -o <out.bin>")?;
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let arch = gguf.req_str("general.architecture")?.to_string();
    let tok = Tokenizer::from_gguf(&gguf)?;
    let input_ids = resolve_prompt_ids(&opts, &tok)?;
    let n_steps = input_ids.len();

    let file = File::create(&out_path)
        .with_context(|| format!("failed to create '{}'", out_path.display()))?;
    let mut writer = BufWriter::new(file);

    if opts.use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        {
            let mut runner = tm_metal::MetalRunner::from_gguf(gguf, opts.max_ctx)?;
            let vocab = runner.vocab_size() as u32;
            writer.write_all(b"TMLG")?;
            writer.write_all(&1u32.to_le_bytes())?;
            writer.write_all(&(n_steps as u32).to_le_bytes())?;
            writer.write_all(&vocab.to_le_bytes())?;
            for &id in &input_ids {
                writer.write_all(&id.to_le_bytes())?;
            }
            for (pos, &id) in input_ids.iter().enumerate() {
                let logits = runner.forward_token(id, pos, true)?;
                write_f32_le(&mut writer, logits)?;
            }
            writer.flush()?;
            eprintln!(
                "[tinymodel logits] wrote {} steps x {} vocab (metal {}) -> {}",
                n_steps,
                vocab,
                arch,
                out_path.display()
            );
            return Ok(());
        }
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
    }

    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();
    let model = CpuModel::from_gguf(&gguf)?;
    check_linux_dirty_budget(model.estimate_dirty_bytes(opts.max_ctx), opts.max_ctx)?;
    let mut state = model.new_state(opts.max_ctx);
    let vocab = model.vocab_size() as u32;
    writer.write_all(b"TMLG")?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&(n_steps as u32).to_le_bytes())?;
    writer.write_all(&vocab.to_le_bytes())?;
    for &id in &input_ids {
        writer.write_all(&id.to_le_bytes())?;
    }
    for (pos, &id) in input_ids.iter().enumerate() {
        model.forward_token(&mut state, &pool, kt, id, pos, true)?;
        write_f32_le(&mut writer, state.logits())?;
    }
    writer.flush()?;
    eprintln!(
        "[tinymodel logits] wrote {} steps x {} vocab (cpu/{} {}) -> {}",
        n_steps,
        vocab,
        kt.isa.name(),
        arch,
        out_path.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        bail!("missing subcommand");
    }

    match args[1].as_str() {
        "probe" => probe::cmd_probe(probe::parse_probe_opts(&args[2..])?),
        "inspect" => {
            let p = args.get(2).context("missing <model.gguf>")?;
            cmd_inspect(PathBuf::from(p))
        }
        "run" => cmd_run(parse_run_opts(&args[2..])?),
        "bench" => cmd_bench(parse_run_opts(&args[2..])?),
        "profile" => cmd_profile(parse_run_opts(&args[2..])?),
        "logits" => cmd_logits(parse_run_opts(&args[2..])?),
        "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => {
            print_usage();
            bail!("unknown subcommand: {other}");
        }
    }
}
