//! `tinymodel` CLI: inspect GGUF models, run greedy inference, and benchmark decode throughput.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{LlamaModel, LlamaState, argmax};
use tm_tokenizer::Tokenizer;

fn print_usage() {
    eprintln!(
        "Usage:
  tinymodel inspect <model.gguf>
  tinymodel run -m <model.gguf> [-p <prompt> | -f <file>] [-n <tokens>] [-t <threads>] [-c <ctx>]
  tinymodel bench -m <model.gguf> [-n <tokens>] [-t <threads>] [-c <ctx>]"
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
}

fn parse_run_opts(args: &[String]) -> Result<RunOpts> {
    let mut model: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_file: Option<PathBuf> = None;
    let mut n_predict = 64usize;
    let mut threads = 4usize;
    let mut max_ctx = 2048usize;

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
    })
}

fn cmd_run(opts: RunOpts) -> Result<()> {
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

    // Prefill prompt tokens
    let t_pp = Instant::now();
    for (idx, &id) in prompt_ids.iter().enumerate() {
        let is_last = idx + 1 == prompt_ids.len();
        model.forward_token(&mut state, &pool, kt, id, idx, is_last)?;
    }
    let pp_sec = t_pp.elapsed().as_secs_f64();

    // Greedy decode
    let mut stdout = io::stdout().lock();
    let mut next_id = argmax(state.logits());
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
        generated += 1;

        if generated >= opts.n_predict {
            break;
        }
        model.forward_token(&mut state, &pool, kt, next_id, pos, true)?;
        next_id = argmax(state.logits());
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
    let gguf = GgufFile::open(&opts.model)?;
    let model = LlamaModel::from_gguf(&gguf)?;
    let mut state = LlamaState::new(&model.cfg, opts.max_ctx);
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();

    // Warmup 2 tokens
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
        "bench: model={} isa={} threads={} tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
        opts.model
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        kt.isa.name(),
        pool.num_threads(),
        n_gen,
        tps,
        (sec * 1e3) / (n_gen as f64),
        tok
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
        "inspect" => {
            let p = args.get(2).context("missing <model.gguf>")?;
            cmd_inspect(PathBuf::from(p))
        }
        "run" => cmd_run(parse_run_opts(&args[2..])?),
        "bench" => cmd_bench(parse_run_opts(&args[2..])?),
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
