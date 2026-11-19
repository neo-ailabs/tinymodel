//! `tinymodel` CLI: inspect GGUF models, run greedy inference, and benchmark decode throughput.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{CpuDecodeProfile, LlamaModel, LlamaState, Sampler, SamplerConfig, argmax};
use tm_tokenizer::Tokenizer;

fn print_usage() {
    eprintln!(
        "Usage:
  tinymodel inspect <model.gguf>
  tinymodel run -m <model.gguf> [-p <prompt> | -f <file>] [-n <tokens>] [-t <threads>] [-c <ctx>] [--metal]
  tinymodel bench -m <model.gguf> [-n <tokens>] [-t <threads>] [-c <ctx>] [--metal]
  tinymodel profile -m <model.gguf> [-n <tokens>] [-t <threads>] [-c <ctx>] [--metal]"
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
    n_predict: usize,
    threads: usize,
    max_ctx: usize,
    use_metal: bool,
    sampler: SamplerConfig,
}

fn parse_run_opts(args: &[String]) -> Result<RunOpts> {
    let mut model: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_file: Option<PathBuf> = None;
    let mut n_predict = 64usize;
    let mut threads = 4usize;
    let mut max_ctx = 2048usize;
    let mut use_metal = false;
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
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    let model = model.context("missing required -m <model.gguf>")?;
    Ok(RunOpts {
        model,
        prompt,
        prompt_file,
        n_predict,
        threads,
        max_ctx,
        use_metal,
        sampler,
    })
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn cmd_run_metal(opts: RunOpts) -> Result<()> {
    let t_load = Instant::now();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let tok = Tokenizer::from_gguf(&gguf)?;
    let mut runner = tm_metal::LlamaMetalRunner::from_gguf(gguf, opts.max_ctx)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let prompt_text = if let Some(ref fpath) = opts.prompt_file {
        std::fs::read_to_string(fpath)
            .with_context(|| format!("failed to read prompt file '{}'", fpath.display()))?
    } else {
        opts.prompt.unwrap_or_else(|| "Hello".to_string())
    };
    let prompt_ids = tok.encode(&prompt_text, tok.add_bos);
    if prompt_ids.is_empty() {
        bail!("prompt encoded to 0 tokens");
    }

    eprintln!(
        "[tinymodel] backend=metal ({}) weights={} load={:.1}ms prompt_tokens={} vocab={} layers={}",
        runner.device_name(),
        if runner.is_zero_copy() {
            "zero-copy"
        } else {
            "copied"
        },
        load_ms,
        prompt_ids.len(),
        runner.cfg.vocab_size,
        runner.cfg.n_layers
    );

    let mut sampler = Sampler::new(opts.sampler, runner.cfg.vocab_size);
    let mut history = prompt_ids.clone();

    let t_pp = Instant::now();
    let logits = runner.forward_batch(&prompt_ids, 0, true)?;
    let mut next_id = sampler.sample(logits, &history);
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
    let mut pos = prompt_ids.len();
    let mut generated = 0usize;

    let is_pure_greedy =
        opts.sampler.temperature <= 0.0 && (opts.sampler.repeat_penalty - 1.0).abs() < 1e-6;
    let t_gen = Instant::now();
    while generated < opts.n_predict {
        if tok.eos_id == Some(next_id) {
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

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn cmd_bench_metal(opts: RunOpts) -> Result<()> {
    let t_load = Instant::now();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let mut runner = tm_metal::LlamaMetalRunner::from_gguf(gguf, opts.max_ctx)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    // Warmup 1 token + 1 small batch
    let _ = runner.forward_token(1, 0, true)?;
    let warmup_ids = [1u32; 8];
    let _ = runner.forward_batch(&warmup_ids, 0, false)?;

    // Measure pp512 (or max_ctx - n_predict - 2 if smaller)
    let n_pp = 512usize.min(opts.max_ctx.saturating_sub(opts.n_predict + 2));
    let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
    let t_pp = Instant::now();
    let _ = runner.forward_batch(&pp_ids, 0, false)?;
    let pp_sec = t_pp.elapsed().as_secs_f64();
    let pp_tps = (n_pp as f64) / pp_sec.max(1e-9);

    // Measure tgN starting from pos 0 using GPU-side greedy argmax
    let mut tok = runner.forward_token_greedy(1, 0)?;
    let n_gen = opts.n_predict;
    let t0 = Instant::now();
    for step in 0..n_gen {
        tok = runner.forward_token_greedy(tok, step + 1)?;
    }
    let sec = t0.elapsed().as_secs_f64();
    let tps = (n_gen as f64) / sec.max(1e-9);
    println!(
        "bench: model={} backend=metal ({}) weights={} load={:.1}ms pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
        opts.model
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        runner.device_name(),
        if runner.is_zero_copy() {
            "zero-copy"
        } else {
            "copied"
        },
        load_ms,
        n_pp,
        pp_tps,
        n_gen,
        tps,
        (sec * 1e3) / (n_gen as f64),
        tok
    );
    Ok(())
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
    let model = LlamaModel::from_gguf(&gguf)?;
    let mut state = LlamaState::new(&model.cfg, opts.max_ctx);
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let prompt_text = if let Some(ref fpath) = opts.prompt_file {
        std::fs::read_to_string(fpath)
            .with_context(|| format!("failed to read prompt file '{}'", fpath.display()))?
    } else {
        opts.prompt.unwrap_or_else(|| "Hello".to_string())
    };

    let prompt_ids = tok.encode(&prompt_text, tok.add_bos);
    if prompt_ids.is_empty() {
        bail!("prompt encoded to 0 tokens");
    }

    eprintln!(
        "[tinymodel] isa={} threads={} load={:.1}ms prompt_tokens={} vocab={} layers={}",
        kt.isa.name(),
        pool.num_threads(),
        load_ms,
        prompt_ids.len(),
        model.cfg.vocab_size,
        model.cfg.n_layers
    );

    // Prefill prompt tokens via batched GEMM
    let t_pp = Instant::now();
    model.forward_batch(&mut state, &pool, kt, &prompt_ids, 0, true)?;
    let pp_sec = t_pp.elapsed().as_secs_f64();

    // Decode with Sampler (temp=0 is greedy argmax fast path)
    let mut stdout = io::stdout().lock();
    let mut sampler = Sampler::new(opts.sampler, model.cfg.vocab_size);
    let mut history = prompt_ids.clone();
    let mut next_id = sampler.sample(state.logits(), &history);
    let mut pos = prompt_ids.len();
    let mut generated = 0usize;

    let t_gen = Instant::now();
    while generated < opts.n_predict {
        if tok.eos_id == Some(next_id) {
            break;
        }
        let bytes = tok.decode_token_bytes(next_id);
        stdout.write_all(&bytes)?;
        stdout.flush()?;
        history.push(next_id);
        generated += 1;

        if generated >= opts.n_predict {
            break;
        }
        model.forward_token(&mut state, &pool, kt, next_id, pos, true)?;
        next_id = sampler.sample(state.logits(), &history);
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
    let gguf = GgufFile::open(&opts.model)?;
    let model = LlamaModel::from_gguf(&gguf)?;
    let mut state = LlamaState::new(&model.cfg, opts.max_ctx);
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();

    // Warmup 1 token + 1 small batch
    model.forward_token(&mut state, &pool, kt, 1, 0, true)?;
    let warm_ids = [1u32; 8];
    model.forward_batch(&mut state, &pool, kt, &warm_ids, 0, false)?;

    // Measure pp512
    let n_pp = 512usize.min(opts.max_ctx.saturating_sub(opts.n_predict + 2));
    let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
    let t_pp = Instant::now();
    model.forward_batch(&mut state, &pool, kt, &pp_ids, 0, false)?;
    let pp_tps = (n_pp as f64) / t_pp.elapsed().as_secs_f64().max(1e-9);

    // Measure tgN starting from pos 0
    model.forward_token(&mut state, &pool, kt, 1, 0, true)?;
    let mut tok = argmax(state.logits());

    let n_gen = opts.n_predict;
    let t0 = Instant::now();
    for step in 0..n_gen {
        model.forward_token(&mut state, &pool, kt, tok, step + 1, true)?;
        tok = argmax(state.logits());
    }
    let sec = t0.elapsed().as_secs_f64();
    let tps = (n_gen as f64) / sec.max(1e-9);
    println!(
        "bench: model={} isa={} threads={} pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
        opts.model
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        kt.isa.name(),
        pool.num_threads(),
        n_pp,
        pp_tps,
        n_gen,
        tps,
        (sec * 1e3) / (n_gen as f64),
        tok
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn cmd_profile_metal(opts: RunOpts) -> Result<()> {
    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?");
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let mut runner = tm_metal::LlamaMetalRunner::from_gguf(gguf, opts.max_ctx)?;
    let n_runs = opts.n_predict.max(16);
    let rep = runner.profile(n_runs, 512)?;

    let dec_tps = 1e6 / rep.decode_single_cb_wall_us.max(1e-3);
    let host_ovh_us = (rep.decode_single_cb_wall_us - rep.decode_single_cb_gpu_us).max(0.0);
    let sum_dec_gpu_us: f64 = rep.decode_stages.iter().map(|s| s.gpu_us).sum();
    let total_dec_bytes: u64 = rep.decode_stages.iter().map(|s| s.bytes).sum();
    let e2e_bw_gbs = (total_dec_bytes as f64 / 1e9) / (rep.decode_single_cb_wall_us * 1e-6);

    println!(
        "=== Metal Profile: {model_name} ({}) ===",
        runner.device_name()
    );
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
        "{:<20} {:>10} {:>8} {:>10} {:>12} {:>14}",
        "Decode Stage", "GPU us", "Share%", "MiB/tok", "Eff GB/s", "Eff GFLOP/s"
    );
    println!("{}", "-".repeat(78));
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
            "{:<20} {:>10.1} {:>7.1}% {:>10.2} {:>12.2} {:>14.1}",
            st.name, st.gpu_us, pct, mib, gbs, gflops
        );
    }
    println!(
        "{:<20} {:>10.1} {:>7.1}% {:>10.2}",
        "Sum (multi-CB)",
        sum_dec_gpu_us,
        100.0,
        total_dec_bytes as f64 / (1024.0 * 1024.0)
    );

    let pp_tps = (rep.prefill_tokens as f64) / (rep.prefill_wall_ms * 1e-3).max(1e-9);
    let sum_pp_gpu_us: f64 = rep.prefill_chunk_stages.iter().map(|s| s.gpu_us).sum();
    println!();
    println!(
        "Prefill pp{}: wall={:.2} ms ({:.1} tok/s) | profiled chunk B={} (gpu_sum={:.2} ms/chunk):",
        rep.prefill_tokens,
        rep.prefill_wall_ms,
        pp_tps,
        rep.prefill_chunk_tokens,
        sum_pp_gpu_us * 1e-3
    );
    println!(
        "{:<20} {:>10} {:>8} {:>10} {:>12} {:>14}",
        "Prefill Stage", "GPU ms", "Share%", "MiB", "Eff GB/s", "Eff GFLOP/s"
    );
    println!("{}", "-".repeat(78));
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
            "{:<20} {:>10.3} {:>7.1}% {:>10.2} {:>12.2} {:>14.1}",
            st.name,
            st.gpu_us * 1e-3,
            pct,
            mib,
            gbs,
            gflops
        );
    }
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
    let model = LlamaModel::from_gguf(&gguf)?;
    let mut state = LlamaState::new(&model.cfg, opts.max_ctx);
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();

    // Warmup
    for i in 0..4 {
        model.forward_token(&mut state, &pool, kt, 1, i, true)?;
    }

    // Measure CPU batched prefill (64 tokens, compute last logits)
    let n_pp = 64usize.min(opts.max_ctx / 2);
    let pp_ids = vec![1u32; n_pp];
    let t_pp = Instant::now();
    model.forward_batch(&mut state, &pool, kt, &pp_ids, 0, true)?;
    let pp_ms = t_pp.elapsed().as_secs_f64() * 1e3;
    let pp_tps = (n_pp as f64) / (pp_ms * 1e-3).max(1e-9);

    // Measure profiled decode over `n_runs` tokens starting at pos 64 (realistic KV context)
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        bail!("missing subcommand");
    }

    match args[1].as_str() {
        "inspect" => {
            let p = args.get(2).context("missing <model.gguf>")?;
            cmd_inspect(PathBuf::from(p))
        }
        "run" => cmd_run(parse_run_opts(&args[2..])?),
        "bench" => cmd_bench(parse_run_opts(&args[2..])?),
        "profile" => cmd_profile(parse_run_opts(&args[2..])?),
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
