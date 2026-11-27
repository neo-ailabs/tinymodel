#![forbid(unsafe_code)]

//! `tinymodel` CLI: inspect GGUF models, probe hardware/telemetry, run inference,
//! dump multi-token logits, benchmark pp/tg throughput, and profile CPU / Metal GPU execution.

mod probe;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tm_core::{DType, KvType, probe_process_memory_stats};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::{ConvertOptions, GgufFile};
use tm_lora::{LoraAdapter, LoraMergeOptions, LoraSet, merge_lora_gguf, parse_lora_spec};
use tm_models::{
    CpuDecodeProfile, Inkling2DecodeProfile, Inkling2Model, Inkling2State, LlamaModel, LlamaState,
    Qwen35DecodeProfile, Qwen35Model, Qwen35State, Sampler, SamplerConfig, argmax,
};
use tm_runtime::{
    CpuModel, LinearMtpHead, MtpDrafter, MultimodalChunk, RgbImage, Session, SpecController,
    SpecMode, SuffixDrafter, VisionEmbeddings, encode_image_staged,
};
use tm_tokenizer::Tokenizer;

fn print_usage() {
    eprintln!(
        "Usage:
  tinymodel probe [-t <threads>] [--stream-mib <mib>] [--no-stream] [--vulkan] [--cuda] [--json]
  tinymodel inspect <model.gguf>
  tinymodel convert -i <in.gguf> -o <out.gguf> [--quant q4_0x4|q4_0|q8_0|iq4_nl|iq3_xxs|q2_0|iq2_xxs|tq2_0] [--lm-head-quant <fmt>] [--prune-vocab <N>] [--vocab-corpus <file>]
  tinymodel lora-merge -m <base.gguf> --lora <adapter.gguf>[:scale] -o <out.gguf> [--quant <fmt>] [--lm-head-quant <fmt>]
  tinymodel run -m <model.gguf> [--mmproj <mmproj.gguf> --image <img.ppm|bmp>] [-md <dspark.gguf>] [--draft-max <k>] [-p <prompt> | -f <file>] [-n <tokens>] [-t <threads>] [-c <ctx>] [--kv-type f16|q8_0] [--spec off|auto|suffix|mtp] [--mtp <mtp.gguf>] [--lora <adapter.gguf>[:scale]] [--cuda | --vulkan | --metal | --gpu <cuda|vulkan|metal|cpu>]
  tinymodel bench -m <model.gguf> [-md <dspark.gguf>] [--draft-max <k>] [-p <prompt> | -f <file>] [-n <tokens>] [-t <threads>] [-c <ctx>] [--kv-type f16|q8_0] [--spec off|auto|suffix|mtp] [--mtp <mtp.gguf>] [--lora <adapter.gguf>[:scale]] [--cuda | --vulkan | --metal | --gpu <cuda|vulkan|metal|cpu>] [--json]
  tinymodel profile -m <model.gguf> [-n <tokens>] [-t <threads>] [-c <ctx>] [--lora <adapter.gguf>[:scale]] [--cuda | --vulkan | --metal | --gpu <cuda|vulkan|metal|cpu>]
  tinymodel logits -m <model.gguf> -o <out.bin> [-p <prompt> | --tokens 1,2,3] [-t <threads>] [-c <ctx>] [--kv-type f16|q8_0] [--lora <adapter.gguf>[:scale]] [--cuda | --vulkan | --metal | --gpu <cuda|vulkan|metal|cpu>]
  tinymodel train-mtp -m <base.gguf> --data <path> -o <mtp.gguf> [--steps <N>] [--seq-len <T>] [--lr <LR>] [--quant f16|q8_0|q4_0|f32] [-t <threads>] [--max-temp-c <C>] [--json]
  tinymodel finetune -m <base.gguf> --data <path> -o <adapter.gguf> [--rank <R>] [--alpha <A>] [--targets q,k,v,o,gate,up,down] [--steps <N>] [--seq-len <T>] [--grad-accum <M>] [--lr <LR>] [--lora-plus-ratio <K>] [--save-every <S>] [--ckpt <ckpt.gguf>] [--resume <ckpt.gguf>] [--f16] [--vulkan | --metal | --gpu <vulkan|metal|cpu>] [--gpu-min-macs <N>] [-t <threads>] [--max-temp-c <C>] [--json]"
    );
}

fn parse_convert_dtype(flag: &str, raw: &str) -> Result<DType> {
    let dt = DType::from_cli_name(raw).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported {flag} '{raw}' (expected q4_0x4, q4_0, q8_0, iq4_nl, iq3_xxs, q2_0, iq2_xxs, tq2_0, f16, f32)"
        )
    })?;
    if !matches!(
        dt,
        DType::Q4_0x4
            | DType::Q4_0
            | DType::Q8_0
            | DType::Iq4Nl
            | DType::Iq3Xxs
            | DType::Q2_0
            | DType::Iq2Xxs
            | DType::TQ2_0
            | DType::F16
            | DType::F32
    ) {
        bail!("unsupported target quantization {flag} '{raw}'");
    }
    Ok(dt)
}

fn cmd_convert(args: &[String]) -> Result<()> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut opts = ConvertOptions::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--input" | "-m" | "--model" => {
                i += 1;
                input = Some(PathBuf::from(
                    args.get(i).context("missing value for -i/--input")?,
                ));
            }
            "-o" | "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    args.get(i).context("missing value for -o/--output")?,
                ));
            }
            "-q" | "--quant" => {
                i += 1;
                let raw = args.get(i).context("missing value for --quant")?;
                opts.quant = Some(parse_convert_dtype("--quant", raw)?);
            }
            "--lm-head-quant" => {
                i += 1;
                let raw = args.get(i).context("missing value for --lm-head-quant")?;
                opts.lm_head_quant = Some(parse_convert_dtype("--lm-head-quant", raw)?);
            }
            "--prune-vocab" => {
                i += 1;
                opts.prune_vocab = Some(
                    args.get(i)
                        .context("missing value for --prune-vocab")?
                        .parse()
                        .context("invalid --prune-vocab")?,
                );
            }
            "--vocab-corpus" => {
                i += 1;
                let cpath = args.get(i).context("missing value for --vocab-corpus")?;
                opts.vocab_corpus = Some(
                    std::fs::read_to_string(cpath)
                        .with_context(|| format!("failed to read --vocab-corpus '{cpath}'"))?,
                );
            }
            other => bail!("unknown convert argument: {other}"),
        }
        i += 1;
    }
    let in_path = input.context("convert requires -i <in.gguf>")?;
    let out_path = output.context("convert requires -o <out.gguf>")?;
    let t0 = Instant::now();
    let gguf = GgufFile::open(&in_path)?;
    let rep = gguf.convert_with_options(&out_path, &opts)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let in_mib = rep.input_file_bytes as f64 / (1024.0 * 1024.0);
    let out_mib = rep.output_file_bytes as f64 / (1024.0 * 1024.0);
    let pct = 100.0 * out_mib / in_mib.max(1e-9);
    if rep.old_vocab_size != rep.new_vocab_size {
        println!(
            "convert: {} -> {} | converted {}/{} tensors | vocab {} -> {} (merges {} -> {}) | {:.2} MiB -> {:.2} MiB ({:.1}%) in {:.1} ms",
            in_path.display(),
            out_path.display(),
            rep.converted_tensors,
            rep.total_tensors,
            rep.old_vocab_size,
            rep.new_vocab_size,
            rep.old_merges,
            rep.kept_merges,
            in_mib,
            out_mib,
            pct,
            ms
        );
    } else {
        println!(
            "convert: {} -> {} | converted {}/{} tensors | {:.2} MiB -> {:.2} MiB ({:.1}%) in {:.1} ms",
            in_path.display(),
            out_path.display(),
            rep.converted_tensors,
            rep.total_tensors,
            in_mib,
            out_mib,
            pct,
            ms
        );
    }
    Ok(())
}

fn cmd_inspect(path: PathBuf) -> Result<()> {
    let gguf = GgufFile::open(&path)?;
    println!("File: {}", gguf.path().display());
    println!(
        "GGUF version: {}, alignment: {} bytes",
        gguf.version(),
        gguf.alignment()
    );
    if let Some(gtype) = gguf.get_meta("general.type").and_then(|v| v.as_str()) {
        println!("Type: {gtype}");
    }
    println!("Tensors: {}", gguf.tensor_names().len());
    if let Ok(arch) = gguf.req_str("general.architecture") {
        println!("Architecture: {arch}");
    }
    if let Some(alpha) = gguf.get_meta("adapter.lora.alpha").and_then(|v| v.as_f32()) {
        println!("LoRA alpha: {alpha}");
    }
    if let Some(d) = gguf.get_meta("mtp.n_embd").and_then(|v| v.as_usize()) {
        println!("MTP n_embd: {d}");
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

fn load_lora_set(specs: &[(PathBuf, f32)]) -> Result<LoraSet> {
    let mut set = LoraSet::new();
    for (p, scale) in specs {
        let adapter = LoraAdapter::open(p)
            .with_context(|| format!("failed to load LoRA adapter '{}'", p.display()))?;
        set.push(adapter, *scale);
    }
    Ok(set)
}

fn cmd_lora_merge(args: &[String]) -> Result<()> {
    let mut base_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut lora_specs: Vec<(PathBuf, f32)> = Vec::new();
    let mut opts = LoraMergeOptions::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" | "-i" | "--input" => {
                i += 1;
                base_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -m/--model")?,
                ));
            }
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -o/--output")?,
                ));
            }
            "--lora" => {
                i += 1;
                let raw = args.get(i).context("missing value for --lora")?;
                let (p, sc) = parse_lora_spec(raw).map_err(|e| anyhow::anyhow!(e))?;
                lora_specs.push((PathBuf::from(p), sc));
            }
            "-q" | "--quant" => {
                i += 1;
                let raw = args.get(i).context("missing value for --quant")?;
                opts.quant = Some(parse_convert_dtype("--quant", raw)?);
            }
            "--lm-head-quant" => {
                i += 1;
                let raw = args.get(i).context("missing value for --lm-head-quant")?;
                opts.lm_head_quant = Some(parse_convert_dtype("--lm-head-quant", raw)?);
            }
            other => bail!("unknown lora-merge argument: {other}"),
        }
        i += 1;
    }

    let base_p = base_path.context("lora-merge requires -m <base.gguf>")?;
    let out_p = out_path.context("lora-merge requires -o <out.gguf>")?;
    if lora_specs.is_empty() {
        bail!("lora-merge requires at least one --lora <adapter.gguf>[:scale]");
    }
    let t0 = Instant::now();
    let base_gguf = GgufFile::open(&base_p)?;
    let loras = load_lora_set(&lora_specs)?;
    let rep = merge_lora_gguf(&base_gguf, &loras, &out_p, &opts)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "lora-merge: {} + {} adapter(s) -> {} | merged {}/{} tensors{} | {:.2} MiB in {:.1} ms",
        base_p.display(),
        lora_specs.len(),
        out_p.display(),
        rep.merged_tensors,
        rep.total_tensors,
        if rep.untied_output {
            " (untied output.weight)"
        } else {
            ""
        },
        rep.output_file_bytes as f64 / (1024.0 * 1024.0),
        ms,
    );
    Ok(())
}

struct RunOpts {
    model: PathBuf,
    draft_model: Option<PathBuf>,
    draft_max: usize,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    explicit_tokens: Option<Vec<u32>>,
    out_path: Option<PathBuf>,
    n_predict: usize,
    threads: usize,
    max_ctx: usize,
    prefix_cache_mib: usize,
    kv_type: KvType,
    spec_mode: SpecMode,
    mtp_path: Option<PathBuf>,
    lora_specs: Vec<(PathBuf, f32)>,
    use_metal: bool,
    use_vulkan: bool,
    use_cuda: bool,
    json: bool,
    sampler: SamplerConfig,
    mmproj_path: Option<PathBuf>,
    image_paths: Vec<PathBuf>,
}

fn parse_run_opts(args: &[String]) -> Result<RunOpts> {
    let mut model: Option<PathBuf> = None;
    let mut draft_model: Option<PathBuf> = None;
    let mut draft_max = 3usize;
    let mut prompt: Option<String> = None;
    let mut prompt_file: Option<PathBuf> = None;
    let mut explicit_tokens: Option<Vec<u32>> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut n_predict = 64usize;
    let mut threads = tm_cpu::recommended_threads();
    let mut max_ctx = 2048usize;
    let mut prefix_cache_mib = 0usize;
    let mut kv_type = KvType::F16;
    let mut spec_mode = SpecMode::Off;
    let mut mtp_path: Option<PathBuf> = None;
    let mut lora_specs: Vec<(PathBuf, f32)> = Vec::new();
    let mut use_metal = false;
    let mut use_vulkan = false;
    let mut use_cuda = false;
    let mut json = false;
    let mut sampler = SamplerConfig::default();
    let mut mmproj_path: Option<PathBuf> = None;
    let mut image_paths: Vec<PathBuf> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model = Some(PathBuf::from(
                    args.get(i).context("missing value for -m/--model")?,
                ));
            }
            "-md" | "--draft" | "--model-draft" => {
                i += 1;
                draft_model = Some(PathBuf::from(
                    args.get(i).context("missing value for -md/--draft")?,
                ));
            }
            "--draft-max" | "--spec-draft-n-max" => {
                i += 1;
                draft_max = args
                    .get(i)
                    .context("missing value for --draft-max")?
                    .parse::<usize>()
                    .context("invalid --draft-max")?
                    .clamp(1, 7);
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
                use_vulkan = false;
                use_cuda = false;
            }
            "--vulkan" => {
                use_vulkan = true;
                use_metal = false;
                use_cuda = false;
            }
            "--cuda" => {
                use_cuda = true;
                use_metal = false;
                use_vulkan = false;
            }
            "--gpu" => {
                i += 1;
                let bk = args.get(i).context("missing value for --gpu")?;
                match bk.as_str() {
                    "cuda" | "nv" => {
                        use_cuda = true;
                        use_vulkan = false;
                        use_metal = false;
                    }
                    "vulkan" | "vk" => {
                        use_vulkan = true;
                        use_metal = false;
                        use_cuda = false;
                    }
                    "metal" | "mtl" => {
                        use_metal = true;
                        use_vulkan = false;
                        use_cuda = false;
                    }
                    "cpu" | "none" => {
                        use_vulkan = false;
                        use_metal = false;
                        use_cuda = false;
                    }
                    other => {
                        bail!("unknown --gpu backend '{other}' (expected cuda, vulkan, metal, cpu)")
                    }
                }
            }
            "--kv-type" => {
                i += 1;
                let raw = args.get(i).context("missing value for --kv-type")?;
                kv_type = raw.parse::<KvType>().map_err(|e| anyhow::anyhow!(e))?;
            }
            "--spec" => {
                i += 1;
                let raw = args.get(i).context("missing value for --spec")?;
                spec_mode = raw.parse::<SpecMode>().map_err(|e| anyhow::anyhow!(e))?;
            }
            "--mtp" => {
                i += 1;
                mtp_path = Some(PathBuf::from(
                    args.get(i).context("missing value for --mtp")?,
                ));
            }
            "--lora" => {
                i += 1;
                let raw = args.get(i).context("missing value for --lora")?;
                let (p, sc) = parse_lora_spec(raw).map_err(|e| anyhow::anyhow!(e))?;
                lora_specs.push((PathBuf::from(p), sc));
            }
            "--prefix-cache-mib" => {
                i += 1;
                prefix_cache_mib = args
                    .get(i)
                    .context("missing value for --prefix-cache-mib")?
                    .parse()
                    .context("invalid --prefix-cache-mib")?;
            }
            "--mmproj" => {
                i += 1;
                mmproj_path = Some(PathBuf::from(
                    args.get(i).context("missing value for --mmproj")?,
                ));
            }
            "--image" => {
                i += 1;
                image_paths.push(PathBuf::from(
                    args.get(i).context("missing value for --image")?,
                ));
            }
            "--json" => {
                json = true;
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    if usize::from(use_metal) + usize::from(use_vulkan) + usize::from(use_cuda) > 1 {
        bail!("cannot specify more than one GPU backend (--cuda, --vulkan, --metal)");
    }
    if !image_paths.is_empty() && mmproj_path.is_none() {
        bail!("--image requires --mmproj <mmproj.gguf>");
    }

    let model = model.context("missing required -m <model.gguf>")?;
    Ok(RunOpts {
        model,
        draft_model,
        draft_max,
        prompt,
        prompt_file,
        explicit_tokens,
        out_path,
        n_predict,
        threads,
        max_ctx,
        prefix_cache_mib,
        kv_type,
        spec_mode,
        mtp_path,
        lora_specs,
        use_metal,
        use_vulkan,
        use_cuda,
        json,
        sampler,
        mmproj_path,
        image_paths,
    })
}

/// Staged multimodal image encoding helper (`ADR-019`): encodes all `--image` files through
/// `--mmproj` and immediately drops the vision encoder mmap before LLM inference starts.
fn encode_cli_images(opts: &RunOpts, pool: &ThreadPool) -> Result<Vec<VisionEmbeddings>> {
    if opts.image_paths.is_empty() {
        return Ok(Vec::new());
    }
    let mmproj = opts
        .mmproj_path
        .as_ref()
        .context("--image requires --mmproj <mmproj.gguf>")?;
    let kt = active_kernels();
    let mut out = Vec::with_capacity(opts.image_paths.len());
    for img_path in &opts.image_paths {
        let t0 = Instant::now();
        let img = RgbImage::from_file(img_path)?;
        let vis = encode_image_staged(mmproj, &img, pool, kt)?;
        eprintln!(
            "[tinymodel] vision: encoded '{}' ({}x{}) -> {} tokens ({}x{}, d={}) in {:.1}ms (staged mmap released)",
            img_path.display(),
            img.width,
            img.height,
            vis.n_tokens,
            vis.grid_h,
            vis.grid_w,
            vis.d_llm,
            t0.elapsed().as_secs_f64() * 1e3
        );
        out.push(vis);
    }
    Ok(out)
}

#[cfg(feature = "vulkan")]
fn encode_cli_images_vulkan(opts: &RunOpts) -> Result<Vec<VisionEmbeddings>> {
    if opts.image_paths.is_empty() {
        return Ok(Vec::new());
    }
    let mmproj = opts
        .mmproj_path
        .as_ref()
        .context("--image requires --mmproj <mmproj.gguf>")?;
    let gguf = std::sync::Arc::new(GgufFile::open(mmproj)?);
    let vis_cfg = tm_models::VisionConfig::from_gguf(&gguf)?;
    let mut runner = match tm_vulkan::VisionVkRunner::from_gguf(gguf) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[tinymodel] vision: Vulkan init fallback to CPU ({e})");
            let pool = ThreadPool::new(opts.threads);
            return encode_cli_images(opts, &pool);
        }
    };
    let dev = runner.device_name().to_string();
    let mut out = Vec::with_capacity(opts.image_paths.len());
    for img_path in &opts.image_paths {
        let t0 = Instant::now();
        let img = RgbImage::from_file(img_path)?;
        let pre = img.preprocess(&vis_cfg);
        let vk_emb = runner.encode_preprocessed(&pre.pixels_chw, pre.width, pre.height)?;
        eprintln!(
            "[tinymodel] vision: encoded '{}' ({}x{}) on Vulkan ({}) -> {} tokens ({}x{}, d={}) in {:.1}ms (staged VRAM released)",
            img_path.display(),
            img.width,
            img.height,
            dev,
            vk_emb.n_tokens,
            vk_emb.grid_h,
            vk_emb.grid_w,
            vk_emb.d_llm,
            t0.elapsed().as_secs_f64() * 1e3
        );
        out.push(VisionEmbeddings {
            embeddings: vk_emb.embeddings,
            n_tokens: vk_emb.n_tokens,
            d_llm: vk_emb.d_llm,
            grid_h: vk_emb.grid_h,
            grid_w: vk_emb.grid_w,
            mrope_positions: vk_emb.mrope_positions,
        });
    }
    // Drop `runner` here before LLM VRAM allocation (staged VLM lifecycle).
    drop(runner);
    Ok(out)
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
    if !opts.lora_specs.is_empty() {
        let loras = load_lora_set(&opts.lora_specs)?;
        runner.bind_lora(&loras)?;
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[tinymodel] arch={} backend=metal ({}) weights={} loras={} load={:.1}ms prompt_tokens={} vocab={} layers={}",
        runner.arch(),
        runner.device_name(),
        if runner.is_zero_copy_weights() {
            "zero-copy"
        } else {
            "copied"
        },
        opts.lora_specs.len(),
        load_ms,
        prompt_ids.len(),
        runner.vocab_size(),
        runner.n_layers()
    );

    let pool = ThreadPool::new(opts.threads);
    let vision_embeds = encode_cli_images(&opts, &pool)?;

    let mut sampler = Sampler::new(opts.sampler, runner.vocab_size());
    let mut history = prompt_ids.clone();

    let t_pp = Instant::now();
    let mut pos = 0usize;
    let total_vis: usize = vision_embeds.iter().map(|v| v.n_tokens).sum();
    let total_pp = total_vis + prompt_ids.len();
    for vis in &vision_embeds {
        for idx in 0..vis.n_tokens {
            let is_last = pos + 1 == total_pp;
            let _ = runner.forward_embed(vis.token_embed(idx), pos, is_last)?;
            pos += 1;
        }
    }
    let logits = if !prompt_ids.is_empty() {
        let l = runner.forward_batch(&prompt_ids, pos, true)?;
        pos += prompt_ids.len();
        l
    } else {
        bail!("prompt must contain at least 1 text token");
    };
    let mut next_id = sampler.sample(logits, &history);
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
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
    let arch = gguf.req_str("general.architecture").unwrap_or("");
    let is_inkling2 = arch == "inkling2" || arch == "inkling2_loop";
    let raw_blocks = gguf
        .get_meta(&format!("{arch}.block_count"))
        .and_then(|v| v.as_usize())
        .unwrap_or(usize::MAX);
    let nextn_layers = gguf
        .get_meta(&format!("{arch}.nextn_predict_layers"))
        .and_then(|v| v.as_usize())
        .unwrap_or(0);
    let n_trunk_layers = raw_blocks.saturating_sub(nextn_layers);
    let gen_n_exp = gguf
        .get_meta(&format!("{arch}.expert_count"))
        .and_then(|v| v.as_usize())
        .unwrap_or(1)
        .max(1);
    let gen_n_exp_used = gguf
        .get_meta(&format!("{arch}.expert_used_count"))
        .and_then(|v| {
            v.as_usize()
                .or_else(|| v.as_usize_array().and_then(|a| a.first().copied()))
        })
        .unwrap_or(gen_n_exp);
    let (core_start, core_end, n_passes, n_exp, n_exp_used) = if is_inkling2 {
        let cs = gguf
            .get_meta(&format!("{arch}.loop.start"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let ce = gguf
            .get_meta(&format!("{arch}.loop.end"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let np = gguf
            .get_meta(&format!("{arch}.loop.count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(1);
        (cs, ce, np, gen_n_exp, gen_n_exp_used)
    } else {
        (0, 0, 1, gen_n_exp, gen_n_exp_used)
    };

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
            if let Some(rest) = name.strip_prefix("blk.")
                && let Some((idx_str, suffix)) = rest.split_once('.')
                && let Ok(ip) = idx_str.parse::<usize>()
            {
                if ip >= n_trunk_layers {
                    continue;
                }
                if is_inkling2 {
                    let in_core = (core_start..core_end).contains(&ip) && n_passes > 1;
                    let mult = if suffix.starts_with("loop_") {
                        n_passes.saturating_sub(1) as u64
                    } else if in_core {
                        n_passes as u64
                    } else {
                        1
                    };
                    if suffix.ends_with("_exps.weight") || suffix.ends_with("_exps.scale") {
                        let per_exp = (info.byte_len / n_exp) as u64;
                        total += mult * (n_exp_used as u64) * per_exp;
                    } else {
                        total += mult * (info.byte_len as u64);
                    }
                    continue;
                }
                if n_exp > 1
                    && (suffix.ends_with("_exps.weight") || suffix.ends_with("_exps.scale"))
                {
                    let per_exp = (info.byte_len / n_exp) as u64;
                    total += (n_exp_used as u64) * per_exp;
                    continue;
                }
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
        let arch = gguf.req_str("general.architecture").unwrap_or("");
        let vocab_cap = gguf
            .get_meta(&format!("{arch}.vocab_size_unpadded"))
            .or_else(|| gguf.get_meta(&format!("{arch}.vocab_size")))
            .and_then(|v| v.as_usize())
            .or_else(|| {
                gguf.tensor_info("token_embd.weight")
                    .and_then(|t| t.dims.get(1).copied())
            })
            .unwrap_or(1001)
            .saturating_sub(1)
            .clamp(1, 1000);
        let n_pp = 512usize.min(opts.max_ctx.saturating_sub(opts.n_predict + 2));
        let ids: Vec<u32> = (0..n_pp).map(|i| ((i % vocab_cap) + 1) as u32).collect();
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
    if !opts.lora_specs.is_empty() {
        let loras = load_lora_set(&opts.lora_specs)?;
        runner.bind_lora(&loras)?;
    }
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
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
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
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = (required_bytes, max_ctx);
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

#[cfg(feature = "vulkan")]
fn print_vulkan_profile_report(
    model_name: &str,
    dev_name: &str,
    drv_info: &str,
    rep: &tm_vulkan::VkProfileReport,
) {
    let dec_tps = 1e6 / rep.decode_single_cb_wall_us.max(1e-3);
    let host_ovh_us = (rep.decode_single_cb_wall_us - rep.decode_single_cb_gpu_us).max(0.0);
    let sum_dec_gpu_us: f64 = rep.decode_stages.iter().map(|s| s.gpu_us).sum();
    let total_dec_bytes: u64 = rep.decode_stages.iter().map(|s| s.bytes).sum();
    let e2e_bw_gbs = (total_dec_bytes as f64 / 1e9) / (rep.decode_single_cb_wall_us * 1e-6);

    println!("=== Vulkan Profile: {model_name} ({dev_name} · {drv_info}) ===");
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
        let gbs = if st.bytes > 0 && st.gpu_us > 0.0 {
            (st.bytes as f64 / 1e9) / (st.gpu_us * 1e-6)
        } else {
            0.0
        };
        let gflops = if st.flops > 0 && st.gpu_us > 0.0 {
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
        "Sum (timestamped)",
        sum_dec_gpu_us,
        100.0,
        total_dec_bytes as f64 / (1024.0 * 1024.0)
    );
    let pp_tps = (rep.prefill_tokens as f64) / (rep.prefill_wall_ms * 1e-3).max(1e-9);
    println!();
    println!(
        "Prefill pp{} ({}): wall={:.2} ms ({:.1} tok/s)",
        rep.prefill_tokens, rep.prefill_backend, rep.prefill_wall_ms, pp_tps
    );
    if !rep.prefill_chunk_stages.is_empty() {
        let sum_pp_us: f64 = rep.prefill_chunk_stages.iter().map(|s| s.gpu_us).sum();
        println!(
            "{:<28} {:>10} {:>8} {:>12} {:>14}",
            "Prefill Stage", "GPU ms", "Share%", "Eff GB/s", "Eff TFLOP/s"
        );
        println!("{}", "-".repeat(76));
        for st in &rep.prefill_chunk_stages {
            let pct = 100.0 * st.gpu_us / sum_pp_us.max(1e-3);
            let gbs = if st.bytes > 0 && st.gpu_us > 0.0 {
                (st.bytes as f64 / 1e9) / (st.gpu_us * 1e-6)
            } else {
                0.0
            };
            let tflops = if st.flops > 0 && st.gpu_us > 0.0 {
                (st.flops as f64 / 1e12) / (st.gpu_us * 1e-6)
            } else {
                0.0
            };
            println!(
                "{:<28} {:>10.2} {:>7.1}% {:>12.2} {:>14.2}",
                st.name,
                st.gpu_us * 1e-3,
                pct,
                gbs,
                tflops
            );
        }
        println!(
            "{:<28} {:>10.2} {:>7.1}%",
            "Sum (timestamped)",
            sum_pp_us * 1e-3,
            100.0
        );
    }
}

#[cfg(feature = "vulkan")]
enum VkDSparkCtx<'a> {
    Gpu(Box<tm_vulkan::DFlashVkRunner>),
    Cpu {
        model: Box<tm_models::DFlashModel<'a>>,
        state: Box<tm_models::DFlashState>,
        taps_buf: Vec<f32>,
        emb_tmp: Vec<f32>,
    },
}

#[cfg(feature = "vulkan")]
impl<'a> VkDSparkCtx<'a> {
    fn from_gguf(
        target: &tm_vulkan::Inkling2VkRunner,
        dg: &'a GgufFile,
        max_ctx: usize,
    ) -> Result<Self> {
        if let Ok(gpu_runner) = tm_vulkan::DFlashVkRunner::from_gguf(target, dg, max_ctx) {
            return Ok(Self::Gpu(Box::new(gpu_runner)));
        }
        let dm = tm_models::DFlashModel::from_gguf(dg)?;
        let ds = tm_models::DFlashState::new(&dm, max_ctx);
        let n_taps = dm.cfg.target_layers.len();
        let n_embd = dm.cfg.n_embd;
        Ok(Self::Cpu {
            model: Box::new(dm),
            state: Box::new(ds),
            taps_buf: vec![0.0; tm_vulkan::MAX_INKLING2_VK_WINDOW * n_taps * n_embd],
            emb_tmp: vec![0.0; n_embd],
        })
    }

    fn target_layers(&self) -> &[usize] {
        match self {
            Self::Gpu(g) => &g.cfg.target_layers,
            Self::Cpu { model, .. } => &model.cfg.target_layers,
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Self::Gpu(g) => g.cfg.block_size,
            Self::Cpu { model, .. } => model.cfg.block_size,
        }
    }

    fn max_draft_tokens(&self) -> usize {
        match self {
            Self::Gpu(g) => g.cfg.max_draft_tokens(),
            Self::Cpu { model, .. } => model.cfg.max_draft_tokens(),
        }
    }

    fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    fn reset(&mut self, target: &tm_vulkan::Inkling2VkRunner) -> Result<()> {
        match self {
            Self::Gpu(g) => Ok(g.reset_state(target)?),
            Self::Cpu { state, .. } => {
                state.reset();
                Ok(())
            }
        }
    }

    fn inject_batch(
        &mut self,
        target: &tm_vulkan::Inkling2VkRunner,
        pool: &ThreadPool,
        kt: &tm_cpu::KernelTable,
        start_pos: usize,
        count: usize,
    ) -> Result<()> {
        match self {
            Self::Gpu(g) => Ok(g.inject_batch(target, start_pos, count)?),
            Self::Cpu {
                model,
                state,
                taps_buf,
                ..
            } => {
                let n_taps = model.cfg.target_layers.len();
                let tap_len = count * n_taps * model.cfg.n_embd;
                target.read_dspark_taps(count, n_taps, &mut taps_buf[..tap_len])?;
                model.inject_batch(state, pool, kt, start_pos, count, &taps_buf[..tap_len])?;
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn inject_and_draft_greedy_with_probs(
        &mut self,
        target: &mut tm_vulkan::Inkling2VkRunner,
        pool: &ThreadPool,
        kt: &tm_cpu::KernelTable,
        inject_start_pos: usize,
        inject_count: usize,
        draft_pos0: usize,
        anchor_token_id: u32,
        max_k: usize,
        out_drafts: &mut [u32],
        out_probs: &mut [f32],
    ) -> Result<usize> {
        match self {
            Self::Gpu(g) => Ok(g.inject_and_draft_greedy_with_probs(
                target,
                inject_start_pos,
                inject_count,
                draft_pos0,
                anchor_token_id,
                max_k,
                out_drafts,
                out_probs,
            )?),
            Self::Cpu {
                model,
                state,
                taps_buf,
                emb_tmp,
            } => {
                if inject_count > 0 {
                    let n_taps = model.cfg.target_layers.len();
                    let tap_len = inject_count * n_taps * model.cfg.n_embd;
                    target.read_dspark_taps(inject_count, n_taps, &mut taps_buf[..tap_len])?;
                    model.inject_batch(
                        state,
                        pool,
                        kt,
                        inject_start_pos,
                        inject_count,
                        &taps_buf[..tap_len],
                    )?;
                }
                if max_k == 0 || draft_pos0 + model.cfg.block_size > state.max_ctx {
                    return Ok(0);
                }
                target.embed_token_normed(anchor_token_id, emb_tmp)?;
                let n = model.draft_block_greedy(
                    state,
                    pool,
                    kt,
                    draft_pos0,
                    anchor_token_id,
                    max_k,
                    |_, out| out.copy_from_slice(emb_tmp),
                    |corr_hid, cap| {
                        target
                            .project_dspark_greedy(corr_hid, cap)
                            .expect("GPU DSpark project_dspark_greedy failed")
                    },
                    &mut out_drafts[..max_k],
                )?;
                out_probs[..n].fill(1.0);
                Ok(n)
            }
        }
    }
}

#[cfg(feature = "vulkan")]
struct DSparkAdaptiveCtrl {
    max_k: usize,
    active_k: usize,
    ema_acc: f32,
    ema_tok_per_step: f64,
    ema_step_s: f64,
    single_step_s: f64,
    spec_steps: usize,
    cooldown: usize,
}

#[cfg(feature = "vulkan")]
impl DSparkAdaptiveCtrl {
    fn new(max_k: usize, single_step_s: f64) -> Self {
        Self {
            max_k,
            active_k: max_k,
            ema_acc: 0.75,
            ema_tok_per_step: 1.0 + 0.75 * (max_k as f64),
            ema_step_s: 0.0,
            single_step_s,
            spec_steps: 0,
            cooldown: 0,
        }
    }

    #[inline]
    fn choose_k(&mut self, cap_k: usize) -> usize {
        if cap_k == 0 || self.max_k == 0 {
            return 0;
        }
        if self.cooldown > 0 {
            self.cooldown -= 1;
            if self.cooldown == 0 {
                // Probe 1 draft token to detect transition back to predictable text/code.
                self.active_k = 1.min(cap_k);
                return self.active_k;
            }
            return 0;
        }
        self.active_k.min(cap_k)
    }

    #[inline]
    fn gate_drafts(drafted: usize, probs: &[f32]) -> usize {
        let mut keep = 0usize;
        let mut cum_p = 1.0f32;
        for (j, &p) in probs.iter().enumerate().take(drafted) {
            cum_p *= p;
            // Calibrated for 201k-vocab `logit_softcap = 30.0` softmax:
            // confident code/boilerplate tokens satisfy `p >= 0.10`, while uncertain open-class
            // prose tails (`p < 0.06..0.10` or cumulative `< 0.03`) are truncated early.
            let ok = if j == 0 {
                p >= 0.06
            } else {
                p >= 0.10 && cum_p >= 0.03
            };
            if !ok {
                break;
            }
            keep = j + 1;
        }
        keep
    }

    fn observe(&mut self, drafted: usize, accepted: usize, committed: usize, step_s: f64) {
        if drafted == 0 {
            if step_s > 0.0 {
                self.single_step_s = if self.single_step_s <= 0.0 {
                    step_s
                } else {
                    0.80 * self.single_step_s + 0.20 * step_s
                };
            }
            return;
        }
        self.spec_steps += 1;
        let step_acc = (accepted as f32) / (drafted as f32);
        self.ema_acc = 0.84 * self.ema_acc + 0.16 * step_acc;
        self.ema_tok_per_step = 0.84 * self.ema_tok_per_step + 0.16 * (committed as f64);
        self.ema_step_s = if self.ema_step_s <= 0.0 {
            step_s
        } else {
            0.84 * self.ema_step_s + 0.16 * step_s
        };

        // If a probe at k=1 succeeded (`accepted == 1`), ramp right back up!
        if self.active_k == 1 && drafted == 1 && accepted == 1 && self.ema_acc < 0.52 {
            self.ema_acc = 0.52;
        }

        if self.spec_steps < 6 {
            return;
        }

        let target_k = if self.ema_acc >= 0.52 {
            self.max_k
        } else if self.ema_acc >= 0.38 {
            self.max_k.min(2)
        } else if self.ema_acc >= 0.24 {
            self.max_k.min(1)
        } else {
            0
        };

        // Profitability guard against measured pure single-token `W=1` latency:
        if self.single_step_s > 0.0 && self.ema_step_s > 0.0 {
            let spec_tps = self.ema_tok_per_step / self.ema_step_s;
            let base_tps = 1.0 / self.single_step_s;
            if spec_tps < 0.94 * base_tps {
                if self.active_k > 1 {
                    self.active_k = (self.active_k - 1).min(target_k.max(1));
                    self.ema_step_s = 0.0;
                    return;
                } else {
                    self.active_k = 0;
                    self.cooldown = 10;
                    self.ema_step_s = 0.0;
                    return;
                }
            }
        }

        self.active_k = target_k;
        if self.active_k == 0 {
            self.cooldown = 10;
        }
    }
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
fn eval_inkling2_vk_window<'a>(
    runner: &'a mut tm_vulkan::Inkling2VkRunner,
    host_helper: &mut Option<(tm_models::Inkling2Model<'_>, tm_models::Inkling2State)>,
    pool: &ThreadPool,
    kt: &tm_cpu::KernelTable,
    window_tokens: &[u32],
    start_pos: usize,
    compute_last_logits: bool,
    save_conv_snapshots: bool,
    dspark_visits: &[usize],
    out_argmax: Option<&mut [u32]>,
) -> Result<&'a [f32]> {
    if let Some((hm, hs)) = host_helper {
        let mut cb = |ip: usize,
                      w_len: usize,
                      cur_win: &[f32],
                      t_ids_win: &[usize],
                      t_w_win: &[f32],
                      m_out_win: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch, win_slot_outs) =
                hs.host_moe_window_parts_mut();
            hm.eval_moe_experts_window_f32(
                ip,
                w_len,
                cur_win,
                t_ids_win,
                t_w_win,
                pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                win_slot_outs,
                m_out_win,
            );
        };
        Ok(runner.forward_window_hybrid(
            window_tokens,
            start_pos,
            compute_last_logits,
            save_conv_snapshots,
            dspark_visits,
            Some(&mut cb),
            out_argmax,
        )?)
    } else {
        Ok(runner.forward_window_hybrid(
            window_tokens,
            start_pos,
            compute_last_logits,
            save_conv_snapshots,
            dspark_visits,
            None,
            out_argmax,
        )?)
    }
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)]
fn eval_inkling2_vk_batch<'a>(
    runner: &'a mut tm_vulkan::Inkling2VkRunner,
    host_helper: &mut Option<(tm_models::Inkling2Model<'_>, tm_models::Inkling2State)>,
    mut dspark: Option<&mut VkDSparkCtx<'_>>,
    pool: &ThreadPool,
    kt: &tm_cpu::KernelTable,
    token_ids: &[u32],
    start_pos: usize,
    compute_last_logits: bool,
) -> Result<&'a [f32]> {
    if token_ids.is_empty() {
        return Ok(runner.forward_batch_hybrid(&[], start_pos, false, None)?);
    }
    let max_w = tm_vulkan::MAX_INKLING2_VK_WINDOW;
    let total = token_ids.len();
    let visits: Vec<usize> = match dspark.as_deref() {
        Some(ds) => ds.target_layers().to_vec(),
        None => Vec::new(),
    };
    let mut off = 0usize;
    while off < total {
        let w = (total - off).min(max_w);
        let is_last = off + w == total;
        let _ = eval_inkling2_vk_window(
            runner,
            host_helper,
            pool,
            kt,
            &token_ids[off..off + w],
            start_pos + off,
            compute_last_logits && is_last,
            false,
            &visits,
            None,
        )?;
        if let Some(ds) = dspark.as_deref_mut() {
            ds.inject_batch(runner, pool, kt, start_pos + off, w)?;
        }
        off += w;
    }
    // Re-borrow immutable logits slice from `runner` (`self.logits_host` was populated on `is_last`)
    Ok(runner.forward_batch_hybrid(&[], start_pos + total, false, None)?)
}

#[cfg(feature = "vulkan")]
fn eval_inkling2_vk_token<'a>(
    runner: &'a mut tm_vulkan::Inkling2VkRunner,
    host_helper: &mut Option<(tm_models::Inkling2Model<'_>, tm_models::Inkling2State)>,
    pool: &ThreadPool,
    kt: &tm_cpu::KernelTable,
    token_id: u32,
    pos: usize,
    compute_logits: bool,
) -> Result<&'a [f32]> {
    if let Some((hm, hs)) = host_helper {
        let mut cb = |ip: usize, cur_n: &[f32], t_ids: &[usize], t_w: &[f32], m_out: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch) = hs.host_moe_parts_mut();
            hm.eval_moe_experts_f32(
                ip,
                cur_n,
                t_ids,
                t_w,
                pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                m_out,
            );
        };
        Ok(runner.forward_token_hybrid(token_id, pos, compute_logits, Some(&mut cb))?)
    } else {
        Ok(runner.forward_token(token_id, pos, compute_logits)?)
    }
}

#[cfg(feature = "vulkan")]
fn eval_inkling2_vk_greedy(
    runner: &mut tm_vulkan::Inkling2VkRunner,
    host_helper: &mut Option<(tm_models::Inkling2Model<'_>, tm_models::Inkling2State)>,
    pool: &ThreadPool,
    kt: &tm_cpu::KernelTable,
    token_id: u32,
    pos: usize,
) -> Result<u32> {
    if let Some((hm, hs)) = host_helper {
        let mut cb = |ip: usize, cur_n: &[f32], t_ids: &[usize], t_w: &[f32], m_out: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch) = hs.host_moe_parts_mut();
            hm.eval_moe_experts_f32(
                ip,
                cur_n,
                t_ids,
                t_w,
                pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                m_out,
            );
        };
        Ok(runner.forward_token_greedy_hybrid(token_id, pos, Some(&mut cb))?)
    } else {
        Ok(runner.forward_token_greedy(token_id, pos)?)
    }
}

#[cfg(feature = "vulkan")]
fn cmd_run_vulkan(opts: RunOpts) -> Result<()> {
    let t_load = Instant::now();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let arch = gguf.req_str("general.architecture")?.to_string();
    let tok = Tokenizer::from_gguf(&gguf)?;
    let prompt_ids = resolve_prompt_ids(&opts, &tok)?;
    let is_pure_greedy =
        opts.sampler.temperature <= 0.0 && (opts.sampler.repeat_penalty - 1.0).abs() < 1e-6;

    let loras = if !opts.lora_specs.is_empty() {
        Some(load_lora_set(&opts.lora_specs)?)
    } else {
        None
    };

    if arch == "inkling2" || arch == "inkling2_loop" {
        if loras.is_some() {
            bail!("--lora is not yet supported on {arch} Vulkan runner");
        }
        let pool = ThreadPool::new(opts.threads);
        let kt = active_kernels();
        let draft_gguf = match opts.draft_model {
            Some(ref dp) => Some(GgufFile::open(dp)?),
            None => None,
        };
        let draft_reserve = if draft_gguf.is_some() {
            512 * 1024 * 1024u64
        } else {
            0
        };
        let mut runner = tm_vulkan::Inkling2VkRunner::from_gguf_with_draft_reserve(
            gguf.clone(),
            opts.max_ctx,
            draft_reserve,
        )?;
        let mut dspark_ctx = match draft_gguf {
            Some(ref dg) => Some(VkDSparkCtx::from_gguf(&runner, dg, opts.max_ctx)?),
            None => None,
        };
        let mut host_helper = if runner.has_host_offload_layers() {
            let hm = tm_models::Inkling2Model::from_gguf_with_options(
                &gguf,
                &tm_models::Inkling2LoadOptions {
                    vram_expert_layers: runner.vram_expert_mask().to_vec(),
                    ram_expert_budget_bytes: None,
                    host_moe_only: true,
                },
            )?;
            for ip in 0..hm.cfg.n_layers_phys {
                if let Some((gu_pin, gu_len, dn_pin, dn_len)) = hm.host_importable_expert_pins(ip) {
                    let _ = runner.register_host_moe_layer(ip, gu_pin, gu_len, dn_pin, dn_len);
                }
            }
            let hs = tm_models::Inkling2State::new_host_moe_only(&hm);
            Some((hm, hs))
        } else {
            None
        };
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
        let tier_info = if let Some((ref hm, _)) = host_helper {
            format!(
                "vram_exps={} ram_exps={} nvme_exps={} ({:.1} MiB/tok direct_io={})",
                hm.stats.vram_expert_layers,
                hm.stats.ram_owned_expert_layers + hm.stats.ram_mmap_expert_layers,
                hm.stats.nvme_direct_expert_layers,
                hm.stats.nvme_bytes_per_decode_tok as f64 / 1048576.0,
                hm.stats.direct_io_active
            )
        } else {
            "all-vram".to_string()
        };
        let draft_tag = if let Some(ref ds) = dspark_ctx {
            format!(
                " · dspark-{}(k={})",
                if ds.is_gpu() { "gpu" } else { "cpu" },
                opts.draft_max
            )
        } else {
            String::new()
        };
        eprintln!(
            "[tinymodel] arch={} backend=vulkan ({} · {} · {} · {}{}) cpu_threads={} load={:.1}ms prompt_tokens={} vocab={} slots={}",
            arch,
            runner.device_name(),
            runner.driver_info(),
            runner.weights_mode(),
            tier_info,
            draft_tag,
            pool.num_threads(),
            load_ms,
            prompt_ids.len(),
            runner.cfg.vocab_size,
            runner.cfg.n_layers,
        );

        let mut sampler = Sampler::new(opts.sampler, runner.cfg.vocab_size);
        let mut history = prompt_ids.clone();

        let t_pp = Instant::now();
        let logits = eval_inkling2_vk_batch(
            &mut runner,
            &mut host_helper,
            dspark_ctx.as_mut(),
            &pool,
            kt,
            &prompt_ids,
            0,
            true,
        )?;
        let mut next_id = sampler.sample(logits, &history);
        let pp_sec = t_pp.elapsed().as_secs_f64();

        let mut stdout = io::stdout().lock();
        let mut pos = prompt_ids.len();
        let mut generated = 0usize;
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;
        let mut verify_steps = 0usize;
        let t_gen = Instant::now();

        if is_pure_greedy && let Some(ref mut ds) = dspark_ctx {
            let mut drafts = [0u32; 8];
            let mut draft_probs = [0.0f32; 8];
            let mut win_toks = [0u32; tm_vulkan::MAX_INKLING2_VK_WINDOW];
            let mut win_argmax = [0u32; tm_vulkan::MAX_INKLING2_VK_WINDOW];
            let target_visits = ds.target_layers().to_vec();
            let bs = ds.block_size();
            let max_d_cfg = ds.max_draft_tokens();
            let mut ctrl = DSparkAdaptiveCtrl::new(opts.draft_max.min(max_d_cfg), 0.0);

            // Emit `next_id` (first token predicted from prompt) immediately
            if !tok.is_stop_token(next_id) {
                stdout.write_all(&tok.decode_token_bytes(next_id))?;
                stdout.flush()?;
                history.push(next_id);
                generated += 1;
            }

            let mut pending_inj: Option<(usize, usize)> = None;
            while generated < opts.n_predict && !tok.is_stop_token(next_id) {
                let t_step = Instant::now();
                let rem_gen = opts.n_predict - generated;
                let cap_k = opts
                    .draft_max
                    .min(max_d_cfg)
                    .min(rem_gen.saturating_sub(1))
                    .min(tm_vulkan::MAX_INKLING2_SPEC_WINDOW - 1);
                let ask_k = if pos + bs <= opts.max_ctx {
                    ctrl.choose_k(cap_k)
                } else {
                    0
                };

                let mut tap_slot = 0usize;
                let drafted = if ask_k > 0 {
                    let (inj_pos, inj_cnt) = pending_inj.take().unwrap_or((0, 0));
                    let raw_d = ds.inject_and_draft_greedy_with_probs(
                        &mut runner,
                        &pool,
                        kt,
                        inj_pos,
                        inj_cnt,
                        pos,
                        next_id,
                        ask_k,
                        &mut drafts[..ask_k],
                        &mut draft_probs[..ask_k],
                    )?;
                    DSparkAdaptiveCtrl::gate_drafts(raw_d, &draft_probs[..raw_d])
                } else {
                    if let Some((inj_pos, inj_cnt)) = pending_inj {
                        if inj_cnt >= tm_vulkan::MAX_INKLING2_SPEC_WINDOW {
                            ds.inject_batch(&runner, &pool, kt, inj_pos, inj_cnt)?;
                            pending_inj = None;
                        } else {
                            tap_slot = inj_cnt;
                        }
                    }
                    0
                };

                let w = 1 + drafted;
                win_toks[0] = next_id;
                win_toks[1..w].copy_from_slice(&drafts[..drafted]);

                runner.set_dspark_tap_slot(tap_slot);
                let _ = eval_inkling2_vk_window(
                    &mut runner,
                    &mut host_helper,
                    &pool,
                    kt,
                    &win_toks[..w],
                    pos,
                    false,
                    drafted > 0,
                    &target_visits,
                    Some(&mut win_argmax[..w]),
                )?;
                runner.set_dspark_tap_slot(0);

                let mut accepted = 0usize;
                let mut committed = 1usize;
                let mut stop_now = false;
                for j in 0..=drafted {
                    let target_tok = win_argmax[j];
                    if j < drafted && target_tok == win_toks[j + 1] {
                        accepted += 1;
                        if tok.is_stop_token(target_tok) {
                            committed = j + 1;
                            next_id = target_tok;
                            stop_now = true;
                            break;
                        }
                        stdout.write_all(&tok.decode_token_bytes(target_tok))?;
                        stdout.flush()?;
                        history.push(target_tok);
                        generated += 1;
                        if generated >= opts.n_predict {
                            committed = j + 1;
                            next_id = target_tok;
                            stop_now = true;
                            break;
                        }
                    } else {
                        committed = j + 1;
                        next_id = target_tok;
                        if tok.is_stop_token(target_tok) {
                            stop_now = true;
                        } else if generated < opts.n_predict {
                            stdout.write_all(&tok.decode_token_bytes(target_tok))?;
                            stdout.flush()?;
                            history.push(target_tok);
                            generated += 1;
                        }
                        break;
                    }
                }

                if drafted > 0 {
                    runner.commit_speculative_window(committed)?;
                    total_drafted += drafted;
                    total_accepted += accepted;
                }
                let step_s = t_step.elapsed().as_secs_f64();
                if drafted > 0 {
                    ctrl.observe(drafted, accepted, committed, step_s);
                } else if ask_k == 0 {
                    ctrl.observe(0, 0, 1, step_s);
                }
                verify_steps += 1;
                pending_inj = match pending_inj {
                    Some((inj_pos, inj_cnt)) => Some((inj_pos, inj_cnt + committed)),
                    None => Some((pos, committed)),
                };
                pos += committed;

                if stop_now || generated >= opts.n_predict {
                    break;
                }
            }
            if let Some((inj_pos, inj_cnt)) = pending_inj.take() {
                ds.inject_batch(&runner, &pool, kt, inj_pos, inj_cnt)?;
            }
        } else {
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
                    next_id = eval_inkling2_vk_greedy(
                        &mut runner,
                        &mut host_helper,
                        &pool,
                        kt,
                        next_id,
                        pos,
                    )?;
                } else {
                    let lg = eval_inkling2_vk_token(
                        &mut runner,
                        &mut host_helper,
                        &pool,
                        kt,
                        next_id,
                        pos,
                        true,
                    )?;
                    next_id = sampler.sample(lg, &history);
                }
                pos += 1;
            }
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
        let spec_note = if total_drafted > 0 {
            format!(
                " | dspark: accepted={}/{} ({:.1}%, verify_steps={}, {:.2} tok/step)",
                total_accepted,
                total_drafted,
                100.0 * (total_accepted as f64) / (total_drafted as f64),
                verify_steps,
                (tg_runs as f64) / (verify_steps.max(1) as f64),
            )
        } else {
            String::new()
        };
        eprintln!(
            "[tinymodel] prompt: {:.2} ms / {} tok ({:.2} tok/s) | eval: {:.2} ms / {} runs ({:.2} tok/s){}",
            pp_sec * 1e3,
            prompt_ids.len(),
            pp_tps,
            gen_sec * 1e3,
            tg_runs,
            tg_tps,
            spec_note,
        );
        return Ok(());
    }

    if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
        let vision_embeds = if !opts.image_paths.is_empty() {
            encode_cli_images_vulkan(&opts)?
        } else {
            Vec::new()
        };
        let mut runner =
            tm_vulkan::Qwen35VkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
        if let Some(ref ls) = loras {
            runner.bind_lora(ls)?;
        }
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[tinymodel] arch={} backend=vulkan ({} · {} · {}) loras={} load={:.1}ms prompt_tokens={} vocab={} layers={} (gdn={}, full={})",
            arch,
            runner.device_name(),
            runner.driver_info(),
            runner.weights_mode(),
            opts.lora_specs.len(),
            load_ms,
            prompt_ids.len(),
            runner.cfg.vocab_size,
            runner.cfg.n_layers,
            runner.cfg.n_gdn_layers,
            runner.cfg.n_full_attn_layers,
        );

        let mut sampler = Sampler::new(opts.sampler, runner.cfg.vocab_size);
        let mut history = prompt_ids.clone();

        let t_pp = Instant::now();
        let mut pos = 0usize;
        for vis in &vision_embeds {
            for t in 0..vis.n_tokens {
                runner.forward_embed(vis.token_embed(t), pos, false)?;
                pos += 1;
            }
        }
        let logits = runner.forward_batch(&prompt_ids, pos, true)?;
        pos += prompt_ids.len();
        let mut next_id = sampler.sample(logits, &history);
        let pp_sec = t_pp.elapsed().as_secs_f64();

        let mut stdout = io::stdout().lock();
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
        return Ok(());
    }

    let vision_embeds = if !opts.image_paths.is_empty() {
        encode_cli_images_vulkan(&opts)?
    } else {
        Vec::new()
    };
    let mut runner = tm_vulkan::LlamaVkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
    if let Some(ref ls) = loras {
        runner.bind_lora(ls)?;
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[tinymodel] arch={} backend=vulkan ({} · {} · {}) loras={} load={:.1}ms prompt_tokens={} vocab={} layers={}",
        runner.cfg.arch,
        runner.device_name(),
        runner.driver_info(),
        runner.weights_mode(),
        opts.lora_specs.len(),
        load_ms,
        prompt_ids.len(),
        runner.cfg.vocab_size,
        runner.cfg.n_layers
    );

    let mut sampler = Sampler::new(opts.sampler, runner.cfg.vocab_size);
    let mut history = prompt_ids.clone();

    let t_pp = Instant::now();
    let mut pos = 0usize;
    for vis in &vision_embeds {
        for t in 0..vis.n_tokens {
            runner.forward_embed(vis.token_embed(t), pos, false)?;
            pos += 1;
        }
    }
    let logits = runner.forward_batch(&prompt_ids, pos, true)?;
    pos += prompt_ids.len();
    let mut next_id = sampler.sample(logits, &history);
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
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

#[cfg(feature = "vulkan")]
fn cmd_bench_vulkan(opts: RunOpts) -> Result<()> {
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

    let loras = if !opts.lora_specs.is_empty() {
        Some(load_lora_set(&opts.lora_specs)?)
    } else {
        None
    };

    if arch == "inkling2" || arch == "inkling2_loop" {
        if loras.is_some() {
            bail!("--lora is not yet supported on {arch} Vulkan runner");
        }
        let pool = ThreadPool::new(opts.threads);
        let kt = active_kernels();
        let draft_gguf = match opts.draft_model {
            Some(ref dp) => Some(GgufFile::open(dp)?),
            None => None,
        };
        let draft_reserve = if draft_gguf.is_some() {
            512 * 1024 * 1024u64
        } else {
            0
        };
        let mut runner = tm_vulkan::Inkling2VkRunner::from_gguf_with_draft_reserve(
            gguf.clone(),
            opts.max_ctx,
            draft_reserve,
        )?;
        let mut dspark_ctx = match draft_gguf {
            Some(ref dg) => Some(VkDSparkCtx::from_gguf(&runner, dg, opts.max_ctx)?),
            None => None,
        };
        let mut host_helper = if runner.has_host_offload_layers() {
            let hm = tm_models::Inkling2Model::from_gguf_with_options(
                &gguf,
                &tm_models::Inkling2LoadOptions {
                    vram_expert_layers: runner.vram_expert_mask().to_vec(),
                    ram_expert_budget_bytes: None,
                    host_moe_only: true,
                },
            )?;
            for ip in 0..hm.cfg.n_layers_phys {
                if let Some((gu_pin, gu_len, dn_pin, dn_len)) = hm.host_importable_expert_pins(ip) {
                    let _ = runner.register_host_moe_layer(ip, gu_pin, gu_len, dn_pin, dn_len);
                }
            }
            let hs = tm_models::Inkling2State::new_host_moe_only(&hm);
            Some((hm, hs))
        } else {
            None
        };
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

        // Warmup 1 token + pre-bake steady-state `BakedCb` / `BakedSpecPlan` before timing
        let _ = eval_inkling2_vk_greedy(&mut runner, &mut host_helper, &pool, kt, 1, 0)?;
        let single_tok_s = if let Some(ref ds) = dspark_ctx {
            let tw = Instant::now();
            let _ = eval_inkling2_vk_greedy(&mut runner, &mut host_helper, &pool, kt, 1, 1)?;
            let s_lat = tw.elapsed().as_secs_f64();
            let warm_w = (1 + opts.draft_max.min(ds.max_draft_tokens()))
                .min(tm_vulkan::MAX_INKLING2_SPEC_WINDOW);
            let warm_toks = [1u32; tm_vulkan::MAX_INKLING2_SPEC_WINDOW];
            let mut warm_am = [0u32; tm_vulkan::MAX_INKLING2_SPEC_WINDOW];
            let warm_visits = ds.target_layers().to_vec();
            let _ = eval_inkling2_vk_window(
                &mut runner,
                &mut host_helper,
                &pool,
                kt,
                &warm_toks[..warm_w],
                2,
                false,
                true,
                &warm_visits,
                Some(&mut warm_am[..warm_w]),
            )?;
            s_lat
        } else {
            0.0
        };
        runner.reset_state()?;
        if let Some((_, ref mut hs)) = host_helper {
            hs.reset();
        }
        if let Some(ref mut ds) = dspark_ctx {
            ds.reset(&runner)?;
        }

        runner.reset_prof();
        let t_pp = Instant::now();
        let pp_logits = eval_inkling2_vk_batch(
            &mut runner,
            &mut host_helper,
            dspark_ctx.as_mut(),
            &pool,
            kt,
            &pp_ids,
            0,
            corpus_mode,
        )?;
        let first_tok = if corpus_mode { argmax(pp_logits) } else { 0 };
        let pp_sec = t_pp.elapsed().as_secs_f64();
        let pp_tps = (n_pp as f64) / pp_sec.max(1e-9);
        if let Some(rep) = runner.prof_report() {
            eprintln!("{rep} (prefill)");
        }

        let n_gen = opts.n_predict;
        let mut gen_tokens = Vec::with_capacity(n_gen + 1);
        let (mut tok, mut cur_pos) = if corpus_mode {
            (first_tok, n_pp)
        } else {
            runner.reset_state()?;
            if let Some((_, ref mut hs)) = host_helper {
                hs.reset();
            }
            if let Some(ref mut ds) = dspark_ctx {
                ds.reset(&runner)?;
                let lg0 = eval_inkling2_vk_batch(
                    &mut runner,
                    &mut host_helper,
                    Some(ds),
                    &pool,
                    kt,
                    &[1u32],
                    0,
                    true,
                )?;
                (argmax(lg0), 1)
            } else {
                (
                    eval_inkling2_vk_greedy(&mut runner, &mut host_helper, &pool, kt, 1, 0)?,
                    1,
                )
            }
        };
        gen_tokens.push(tok);

        let mut dspark_accepted = 0usize;
        let mut dspark_drafted = 0usize;
        runner.reset_prof();
        let t0 = Instant::now();
        if let Some(ref mut ds) = dspark_ctx {
            let mut drafts = [0u32; 8];
            let mut draft_probs = [0.0f32; 8];
            let mut win_toks = [0u32; tm_vulkan::MAX_INKLING2_VK_WINDOW];
            let mut win_argmax = [0u32; tm_vulkan::MAX_INKLING2_VK_WINDOW];
            let target_visits = ds.target_layers().to_vec();
            let bs = ds.block_size();
            let max_d_cfg = ds.max_draft_tokens();
            let mut ctrl = DSparkAdaptiveCtrl::new(opts.draft_max.min(max_d_cfg), single_tok_s);
            let mut pending_inj: Option<(usize, usize)> = None;

            while gen_tokens.len() < n_gen + 1 {
                let t_step = Instant::now();
                let rem = (n_gen + 1) - gen_tokens.len();
                let cap_k = opts
                    .draft_max
                    .min(max_d_cfg)
                    .min(rem.saturating_sub(1))
                    .min(tm_vulkan::MAX_INKLING2_SPEC_WINDOW - 1);
                let ask_k = if cur_pos + bs <= opts.max_ctx {
                    ctrl.choose_k(cap_k)
                } else {
                    0
                };

                let mut tap_slot = 0usize;
                let drafted = if ask_k > 0 {
                    let (inj_pos, inj_cnt) = pending_inj.take().unwrap_or((0, 0));
                    let raw_d = ds.inject_and_draft_greedy_with_probs(
                        &mut runner,
                        &pool,
                        kt,
                        inj_pos,
                        inj_cnt,
                        cur_pos,
                        tok,
                        ask_k,
                        &mut drafts[..ask_k],
                        &mut draft_probs[..ask_k],
                    )?;
                    DSparkAdaptiveCtrl::gate_drafts(raw_d, &draft_probs[..raw_d])
                } else {
                    if let Some((inj_pos, inj_cnt)) = pending_inj {
                        if inj_cnt >= tm_vulkan::MAX_INKLING2_SPEC_WINDOW {
                            ds.inject_batch(&runner, &pool, kt, inj_pos, inj_cnt)?;
                            pending_inj = None;
                        } else {
                            tap_slot = inj_cnt;
                        }
                    }
                    0
                };

                let w = 1 + drafted;
                win_toks[0] = tok;
                win_toks[1..w].copy_from_slice(&drafts[..drafted]);

                runner.set_dspark_tap_slot(tap_slot);
                let _ = eval_inkling2_vk_window(
                    &mut runner,
                    &mut host_helper,
                    &pool,
                    kt,
                    &win_toks[..w],
                    cur_pos,
                    false,
                    drafted > 0,
                    &target_visits,
                    Some(&mut win_argmax[..w]),
                )?;
                runner.set_dspark_tap_slot(0);

                let mut accepted = 0usize;
                let mut committed = 1usize;
                for j in 0..=drafted {
                    let target_tok = win_argmax[j];
                    if j < drafted && target_tok == win_toks[j + 1] {
                        accepted += 1;
                        gen_tokens.push(target_tok);
                        tok = target_tok;
                        if gen_tokens.len() > n_gen {
                            committed = j + 1;
                            break;
                        }
                    } else {
                        committed = j + 1;
                        tok = target_tok;
                        if gen_tokens.len() <= n_gen {
                            gen_tokens.push(target_tok);
                        }
                        break;
                    }
                }
                if drafted > 0 {
                    runner.commit_speculative_window(committed)?;
                    dspark_drafted += drafted;
                    dspark_accepted += accepted;
                }
                let step_s = t_step.elapsed().as_secs_f64();
                if drafted > 0 {
                    ctrl.observe(drafted, accepted, committed, step_s);
                } else if ask_k == 0 {
                    ctrl.observe(0, 0, 1, step_s);
                }
                pending_inj = match pending_inj {
                    Some((inj_pos, inj_cnt)) => Some((inj_pos, inj_cnt + committed)),
                    None => Some((cur_pos, committed)),
                };
                cur_pos += committed;
            }
            if let Some((inj_pos, inj_cnt)) = pending_inj.take() {
                ds.inject_batch(&runner, &pool, kt, inj_pos, inj_cnt)?;
            }
        } else {
            for step in 0..n_gen {
                tok = eval_inkling2_vk_greedy(
                    &mut runner,
                    &mut host_helper,
                    &pool,
                    kt,
                    tok,
                    cur_pos + step,
                )?;
                gen_tokens.push(tok);
            }
        }
        let sec = t0.elapsed().as_secs_f64();
        let tps = (n_gen as f64) / sec.max(1e-9);
        if let Some(rep) = runner.prof_report() {
            eprintln!("{rep} (decode)");
        }
        let tel_after = probe::collect_telemetry();
        let dev_name = runner.device_name().to_string();
        let wmode = if let Some(ref ds) = dspark_ctx {
            format!(
                "{}+dspark_{}_k{}({}/{})",
                runner.weights_mode(),
                if ds.is_gpu() { "gpu" } else { "cpu" },
                opts.draft_max,
                dspark_accepted,
                dspark_drafted
            )
        } else {
            runner.weights_mode().to_string()
        };
        if opts.json {
            BenchMetrics {
                model_name: &model_name,
                arch: &arch,
                backend: "vulkan",
                isa: kt.isa.name(),
                device_name: &dev_name,
                weights_mode: &wmode,
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
                "bench: model={} backend=vulkan ({}) weights={} threads={} load={:.1}ms pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
                model_name,
                dev_name,
                wmode,
                pool.num_threads(),
                load_ms,
                n_pp,
                pp_tps,
                n_gen,
                tps,
                (sec * 1e3) / (n_gen as f64),
                tok
            );
        }
        return Ok(());
    }

    if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
        let mut runner =
            tm_vulkan::Qwen35VkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
        if let Some(ref ls) = loras {
            runner.bind_lora(ls)?;
        }
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

        let _ = runner.forward_token_greedy(1, 0)?;
        let t_warm = Instant::now();
        let mut warm_iters = 0usize;
        while warm_iters < 2 || t_warm.elapsed().as_secs_f64() < 0.35 {
            runner.reset_state()?;
            let _ = runner.forward_batch(&pp_ids, 0, false)?;
            warm_iters += 1;
        }

        runner.reset_state()?;
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
            runner.reset_state()?;
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
        let dev_name = runner.device_name().to_string();
        let wmode = runner.weights_mode();
        if opts.json {
            BenchMetrics {
                model_name: &model_name,
                arch: &arch,
                backend: "vulkan",
                isa: "vulkan",
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
                "bench: model={} backend=vulkan ({}) weights={} load={:.1}ms pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
                model_name,
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
        return Ok(());
    }

    let mut runner = tm_vulkan::LlamaVkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
    if let Some(ref ls) = loras {
        runner.bind_lora(ls)?;
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    // Warm up both decode and batched prefill pipelines (>= 2 passes and >= 350 ms, matching
    // `probe.rs` DVFS warmup and `llama-bench` steady-state SMU clock ramp).
    let _ = runner.forward_token_greedy(1, 0)?;
    let t_warm = Instant::now();
    let mut warm_iters = 0usize;
    while warm_iters < 2 || t_warm.elapsed().as_secs_f64() < 0.35 {
        let _ = runner.forward_batch(&pp_ids, 0, false)?;
        warm_iters += 1;
    }

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
    let dev_name = runner.device_name().to_string();
    let wmode = runner.weights_mode();
    if opts.json {
        BenchMetrics {
            model_name: &model_name,
            arch: &arch,
            backend: "vulkan",
            isa: "vulkan",
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
            "bench: model={} backend=vulkan ({}) weights={} load={:.1}ms pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
            model_name,
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

#[cfg(feature = "vulkan")]
fn cmd_profile_vulkan(opts: RunOpts) -> Result<()> {
    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let gguf = std::sync::Arc::new(GgufFile::open(&opts.model)?);
    let arch = gguf.req_str("general.architecture")?.to_string();
    let loras = if !opts.lora_specs.is_empty() {
        Some(load_lora_set(&opts.lora_specs)?)
    } else {
        None
    };
    let n_runs = opts.n_predict.max(16);
    if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
        let mut runner =
            tm_vulkan::Qwen35VkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
        if let Some(ref ls) = loras {
            runner.bind_lora(ls)?;
        }
        let dev = runner.device_name().to_string();
        let drv = runner.driver_info().to_string();
        let rep = runner.profile(n_runs, 512)?;
        print_vulkan_profile_report(&model_name, &dev, &drv, &rep);
        return Ok(());
    }
    let mut runner = tm_vulkan::LlamaVkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
    if let Some(ref ls) = loras {
        runner.bind_lora(ls)?;
    }
    let dev = runner.device_name().to_string();
    let drv = runner.driver_info().to_string();
    let rep = runner.profile(n_runs, 512)?;
    print_vulkan_profile_report(&model_name, &dev, &drv, &rep);
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_run_cuda(opts: RunOpts) -> Result<()> {
    let t_load = Instant::now();
    let gguf = GgufFile::open(&opts.model)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let prompt_ids = resolve_prompt_ids(&opts, &tok)?;
    let is_pure_greedy =
        opts.sampler.temperature <= 0.0 && (opts.sampler.repeat_penalty - 1.0).abs() < 1e-6;

    let mut runner =
        tm_cuda::CudaRunner::from_gguf_with_kv_type(&gguf, opts.max_ctx, opts.kv_type)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let info = runner.device_info().clone();
    eprintln!(
        "[tinymodel] arch={} backend=cuda ({}, sm_{}{}) vram={:.1}MiB kv={} load={:.1}ms prompt_tokens={} vocab={} layers={}",
        runner.arch(),
        info.name,
        info.cc_major,
        info.cc_minor,
        runner.vram_allocated_bytes() as f64 / (1024.0 * 1024.0),
        opts.kv_type.as_str(),
        load_ms,
        prompt_ids.len(),
        runner.vocab_size(),
        runner.n_layers()
    );

    let pool = ThreadPool::new(opts.threads);
    let vision_embeds = encode_cli_images(&opts, &pool)?;

    let mut sampler = Sampler::new(opts.sampler, runner.vocab_size());
    let mut history = prompt_ids.clone();

    let t_pp = Instant::now();
    let mut pos = 0usize;
    let total_vis: usize = vision_embeds.iter().map(|v| v.n_tokens).sum();
    let total_pp = total_vis + prompt_ids.len();
    for vis in &vision_embeds {
        for idx in 0..vis.n_tokens {
            let is_last = pos + 1 == total_pp;
            let _ = runner.forward_embed(vis.token_embed(idx), pos, is_last)?;
            pos += 1;
        }
    }
    let logits = if !prompt_ids.is_empty() {
        let l = runner.forward_batch(&prompt_ids, pos, true)?;
        pos += prompt_ids.len();
        l
    } else {
        bail!("prompt must contain at least 1 text token");
    };
    let mut next_id = sampler.sample(logits, &history);
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
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
    let tg_runs = generated.saturating_sub(1);
    let pp_tps = (total_pp as f64) / pp_sec.max(1e-9);
    let tg_tps = if tg_runs > 0 {
        (tg_runs as f64) / gen_sec.max(1e-9)
    } else {
        0.0
    };
    eprintln!(
        "[tinymodel] timings (cuda): prefill {:.1}ms ({} tok, {:.2} tok/s) | decode {:.1}ms ({} tok, {:.2} tok/s)",
        pp_sec * 1e3,
        total_pp,
        pp_tps,
        gen_sec * 1e3,
        tg_runs,
        tg_tps
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_bench_cuda(opts: RunOpts) -> Result<()> {
    let tel_before = probe::collect_telemetry();
    let t_load = Instant::now();
    let gguf = GgufFile::open(&opts.model)?;
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

    let mut runner =
        tm_cuda::CudaRunner::from_gguf_with_kv_type(&gguf, opts.max_ctx, opts.kv_type)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    // Warmup both MMVQ decode and WMMA Tensor Core prefill paths
    let _ = runner.forward_token_greedy(1, 0)?;
    let _ = runner.forward_token_greedy(2, 1)?;
    runner.reset_state()?;
    let warmup_ids = [1u32; 16];
    let _ = runner.forward_batch(&warmup_ids, 0, true)?;
    runner.reset_state()?;

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
        runner.reset_state()?;
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
    let info = runner.device_info().clone();
    let dev_name = format!("{} (sm_{}{})", info.name, info.cc_major, info.cc_minor);

    if opts.json {
        BenchMetrics {
            model_name: &model_name,
            arch: &arch,
            backend: "cuda",
            isa: "cuda",
            device_name: &dev_name,
            weights_mode: "device-vram",
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
        let eff_bw = (active_decode_bytes as f64 / 1e9) * tps;
        println!(
            "bench: model={} arch={} backend=cuda ({}) vram={:.1}MiB load={:.1}ms pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, bw={:.1}/{:.1} GB/s = {:.1}%, last_tok={})",
            model_name,
            arch,
            dev_name,
            runner.vram_allocated_bytes() as f64 / (1024.0 * 1024.0),
            load_ms,
            n_pp,
            pp_tps,
            n_gen,
            tps,
            (sec * 1e3) / (n_gen as f64),
            eff_bw,
            info.peak_mem_bw_gbps,
            100.0 * eff_bw / info.peak_mem_bw_gbps.max(1.0),
            tok
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cmd_profile_cuda(opts: RunOpts) -> Result<()> {
    let model_name = opts
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let gguf = GgufFile::open(&opts.model)?;
    let n_runs = opts.n_predict.max(32);
    let mut runner =
        tm_cuda::CudaRunner::from_gguf_with_kv_type(&gguf, opts.max_ctx, opts.kv_type)?;
    let rep = runner.profile(n_runs, 512)?;
    println!(
        "=== CUDA GPU Decode Profile: {} ({}, arch={}) ===",
        model_name, rep.device_name, rep.arch
    );
    println!(
        "Decode: {:.3} ms/tok ({:.2} tok/s) | Weights: {:.1} MiB | Effective BW: {:.2} / {:.1} GB/s ({:.1}%)",
        rep.total_ms_per_tok,
        rep.tok_per_sec,
        rep.model_weight_bytes as f64 / (1024.0 * 1024.0),
        rep.effective_bw_gbps,
        rep.peak_bw_gbps,
        rep.bw_utilization_pct
    );
    Ok(())
}

fn cmd_run(opts: RunOpts) -> Result<()> {
    if opts.use_cuda {
        #[cfg(feature = "cuda")]
        return cmd_run_cuda(opts);
        #[cfg(not(feature = "cuda"))]
        bail!("--cuda requires compiling with feature 'cuda'");
    }
    if opts.use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        return cmd_run_metal(opts);
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
    }
    if opts.use_vulkan {
        #[cfg(feature = "vulkan")]
        return cmd_run_vulkan(opts);
        #[cfg(not(feature = "vulkan"))]
        bail!("--vulkan requires compiling with feature 'vulkan'");
    }
    if let Some(ref dp) = opts.draft_model {
        bail!(
            "-md/--draft ({}, --draft-max {}) currently requires --vulkan",
            dp.display(),
            opts.draft_max
        );
    }
    let t_load = Instant::now();
    let gguf = GgufFile::open(&opts.model)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let prompt_ids = resolve_prompt_ids(&opts, &tok)?;
    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();

    let mut model = CpuModel::from_gguf(&gguf)?;
    if !opts.lora_specs.is_empty() {
        let loras = load_lora_set(&opts.lora_specs)?;
        model.bind_lora(&gguf, &loras)?;
    }
    let mut mtp_head = if let Some(ref mp) = opts.mtp_path {
        Some(LinearMtpHead::open(mp, model.arch(), model.n_embd())?)
    } else {
        None
    };
    check_linux_dirty_budget(
        model.estimate_dirty_bytes_with_kv(opts.max_ctx, opts.kv_type),
        opts.max_ctx,
    )?;
    let cache_bytes = opts.prefix_cache_mib * 1024 * 1024;
    let mut session = Session::new_with_options(&model, opts.max_ctx, cache_bytes, opts.kv_type);
    if opts.spec_mode != SpecMode::Off {
        session.enable_speculation();
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "[tinymodel] arch={} isa={} threads={} kv={} spec={} loras={} load={:.1}ms prompt_tokens={} vocab={} layers={} prefix_cache_mib={}",
        model.arch(),
        kt.isa.name(),
        pool.num_threads(),
        opts.kv_type.as_str(),
        opts.spec_mode.as_str(),
        opts.lora_specs.len(),
        load_ms,
        prompt_ids.len(),
        model.vocab_size(),
        model.n_layers(),
        opts.prefix_cache_mib
    );
    if let CpuModel::Inkling2(ref m) = model {
        let st = &m.stats;
        eprintln!(
            "[tinymodel] inkling2 placement: dense={:.1}MiB | ram_mmap_layers={} | ram_owned_layers={} ({:.1}MiB, core_in_fast={}/12) | ssd_stream_layers={} ({:.1}MiB/tok)",
            st.dense_bytes as f64 / 1048576.0,
            st.ram_mmap_expert_layers,
            st.ram_owned_expert_layers,
            st.ram_owned_bytes as f64 / 1048576.0,
            st.core_layers_in_fast_tiers,
            st.nvme_direct_expert_layers,
            st.nvme_bytes_per_decode_tok as f64 / 1048576.0,
        );
    }

    let vision_embeds = encode_cli_images(&opts, &pool)?;
    let t_pp = Instant::now();
    if !vision_embeds.is_empty() {
        let mut chunks: Vec<MultimodalChunk<'_>> = Vec::with_capacity(vision_embeds.len() + 1);
        for vis in &vision_embeds {
            chunks.push(MultimodalChunk::Image(vis));
        }
        chunks.push(MultimodalChunk::Tokens(&prompt_ids));
        let _ = session.prefill_multimodal(&pool, &chunks)?;
    } else {
        let rep =
            session.prefill_with_cache(&pool, &prompt_ids, (cache_bytes > 0).then_some(64))?;
        if cache_bytes > 0 && rep.cached_tokens > 0 {
            eprintln!(
                "[tinymodel] prefix_cache hit: reused {}/{} tokens",
                rep.cached_tokens, rep.total_tokens
            );
        }
    }
    let pp_sec = t_pp.elapsed().as_secs_f64();

    let mut stdout = io::stdout().lock();
    let is_pure_greedy =
        opts.sampler.temperature <= 0.0 && (opts.sampler.repeat_penalty - 1.0).abs() < 1e-6;

    let (generated, gen_sec, spec_note) = if opts.spec_mode != SpecMode::Off && is_pure_greedy {
        let mut suffix = SuffixDrafter::default();
        let mut ctrl = SpecController::default();
        let t_gen = Instant::now();
        let (tokens, stats) = session.generate_greedy_speculative(
            &pool,
            &prompt_ids,
            opts.n_predict,
            opts.spec_mode,
            &mut suffix,
            mtp_head.as_mut().map(|h| h as &mut dyn MtpDrafter),
            &mut ctrl,
            |_, tid| tok.is_stop_token(tid),
        )?;
        let elapsed = t_gen.elapsed().as_secs_f64();
        let mut emitted = 0usize;
        for &tid in &tokens {
            if tok.is_stop_token(tid) {
                break;
            }
            stdout.write_all(&tok.decode_token_bytes(tid))?;
            emitted += 1;
        }
        stdout.flush()?;
        (
            emitted,
            elapsed,
            format!(
                " | spec: accepted={}/{} ({:.1}%, verify_calls={}, auto_off={})",
                stats.accepted_tokens,
                stats.drafted_tokens,
                stats.acceptance_rate() * 100.0,
                stats.verify_calls,
                stats.auto_disabled_steps
            ),
        )
    } else {
        let mut sampler = Sampler::new(opts.sampler, model.vocab_size());
        let mut history = prompt_ids.clone();
        let mut next_id = sampler.sample(session.logits(), &history);
        let mut pos = prompt_ids.len();
        let mut count = 0usize;

        let t_gen = Instant::now();
        while count < opts.n_predict {
            if tok.is_stop_token(next_id) {
                break;
            }
            stdout.write_all(&tok.decode_token_bytes(next_id))?;
            stdout.flush()?;
            history.push(next_id);
            count += 1;

            if count >= opts.n_predict {
                break;
            }
            model.forward_token(&mut session.state, &pool, kt, next_id, pos, true)?;
            next_id = sampler.sample(session.logits(), &history);
            pos += 1;
        }
        (count, t_gen.elapsed().as_secs_f64(), String::new())
    };

    writeln!(stdout)?;
    let pp_tps = (prompt_ids.len() as f64) / pp_sec.max(1e-9);
    let tg_runs = generated.saturating_sub(1);
    let tg_tps = if tg_runs > 0 {
        (tg_runs as f64) / gen_sec.max(1e-9)
    } else {
        0.0
    };
    eprintln!(
        "[tinymodel] prompt: {:.2} ms / {} tok ({:.2} tok/s) | eval: {:.2} ms / {} runs ({:.2} tok/s){}",
        pp_sec * 1e3,
        prompt_ids.len(),
        pp_tps,
        gen_sec * 1e3,
        tg_runs,
        tg_tps,
        spec_note
    );

    Ok(())
}

fn cmd_bench(opts: RunOpts) -> Result<()> {
    if opts.use_cuda {
        #[cfg(feature = "cuda")]
        return cmd_bench_cuda(opts);
        #[cfg(not(feature = "cuda"))]
        bail!("--cuda requires compiling with feature 'cuda'");
    }
    if opts.use_vulkan {
        #[cfg(feature = "vulkan")]
        return cmd_bench_vulkan(opts);
        #[cfg(not(feature = "vulkan"))]
        bail!("--vulkan requires compiling with feature 'vulkan'");
    }
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

    let mut model = CpuModel::from_gguf(&gguf)?;
    if !opts.lora_specs.is_empty() {
        let loras = load_lora_set(&opts.lora_specs)?;
        model.bind_lora(&gguf, &loras)?;
    }
    let mut mtp_head = if let Some(ref mp) = opts.mtp_path {
        Some(LinearMtpHead::open(mp, model.arch(), model.n_embd())?)
    } else {
        None
    };
    check_linux_dirty_budget(
        model.estimate_dirty_bytes_with_kv(opts.max_ctx, opts.kv_type),
        opts.max_ctx,
    )?;
    let mut session = Session::new_with_options(&model, opts.max_ctx, 0, opts.kv_type);
    if opts.spec_mode != SpecMode::Off {
        session.enable_speculation();
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    if let CpuModel::Inkling2(ref m) = model {
        let st = &m.stats;
        eprintln!(
            "[tinymodel] inkling2 placement: dense={:.1}MiB | ram_mmap_layers={} | ram_owned_layers={} ({:.1}MiB, core_in_fast={}/12) | ssd_stream_layers={} ({:.1}MiB/tok)",
            st.dense_bytes as f64 / 1048576.0,
            st.ram_mmap_expert_layers,
            st.ram_owned_expert_layers,
            st.ram_owned_bytes as f64 / 1048576.0,
            st.core_layers_in_fast_tiers,
            st.nvme_direct_expert_layers,
            st.nvme_bytes_per_decode_tok as f64 / 1048576.0,
        );
    }

    // Warm up mmap pages & thread pool
    model.forward_token(&mut session.state, &pool, kt, 1, 0, true)?;
    session.state.reset();
    model.forward_batch(&mut session.state, &pool, kt, &[1u32; 8], 0, false)?;
    session.state.reset();

    let t_pp = Instant::now();
    if opts.spec_mode != SpecMode::Off {
        session.prefill_with_cache(&pool, &pp_ids, None)?;
    } else {
        model.forward_batch(&mut session.state, &pool, kt, &pp_ids, 0, corpus_mode)?;
    }
    let first_tok = if corpus_mode || opts.spec_mode != SpecMode::Off {
        argmax(session.logits())
    } else {
        0
    };
    let pp_sec = t_pp.elapsed().as_secs_f64();
    let pp_tps = (n_pp as f64) / pp_sec.max(1e-9);

    let n_gen = opts.n_predict;
    let mut gen_tokens = Vec::with_capacity(n_gen + 1);
    let mut spec_summary = String::new();
    let (tok, sec) = if opts.spec_mode != SpecMode::Off {
        let mut suffix = SuffixDrafter::default();
        let mut ctrl = SpecController::default();
        if !corpus_mode {
            session.prefill_with_cache(&pool, &[1u32], None)?;
        }
        let seed_prompt: &[u32] = if corpus_mode { &pp_ids } else { &[1u32] };
        let t0 = Instant::now();
        let (toks, stats) = session.generate_greedy_speculative(
            &pool,
            seed_prompt,
            n_gen + 1,
            opts.spec_mode,
            &mut suffix,
            mtp_head.as_mut().map(|h| h as &mut dyn MtpDrafter),
            &mut ctrl,
            |_, _| false,
        )?;
        let elapsed = t0.elapsed().as_secs_f64();
        let last = toks.last().copied().unwrap_or(first_tok);
        gen_tokens = toks;
        spec_summary = format!(
            " spec={}({}/{})",
            opts.spec_mode.as_str(),
            stats.accepted_tokens,
            stats.drafted_tokens
        );
        (last, elapsed)
    } else {
        let (mut cur_tok, base_pos) = if corpus_mode {
            (first_tok, n_pp)
        } else {
            session.state.reset();
            model.forward_token(&mut session.state, &pool, kt, 1, 0, true)?;
            (argmax(session.logits()), 1)
        };
        gen_tokens.push(cur_tok);
        let t0 = Instant::now();
        for step in 0..n_gen {
            model.forward_token(
                &mut session.state,
                &pool,
                kt,
                cur_tok,
                base_pos + step,
                true,
            )?;
            cur_tok = argmax(session.logits());
            gen_tokens.push(cur_tok);
        }
        (cur_tok, t0.elapsed().as_secs_f64())
    };
    let tps = (n_gen as f64) / sec.max(1e-9);
    let tel_after = probe::collect_telemetry();
    let wmode = match (opts.kv_type, opts.spec_mode) {
        (KvType::F16, SpecMode::Off) => "mmap".to_string(),
        (kv, SpecMode::Off) => format!("mmap+kv_{}", kv.as_str()),
        (kv, sp) => format!("mmap+kv_{}+spec_{}", kv.as_str(), sp.as_str()),
    };
    if opts.json {
        BenchMetrics {
            model_name: &model_name,
            arch: &arch,
            backend: "cpu",
            isa: kt.isa.name(),
            device_name: &dev_model_name,
            weights_mode: &wmode,
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
            "bench: model={} arch={} isa={} threads={} kv={}{} pp{}={:.2} tok/s tg{}={:.2} tok/s ({:.2} ms/tok, last_tok={})",
            model_name,
            arch,
            kt.isa.name(),
            pool.num_threads(),
            opts.kv_type.as_str(),
            spec_summary,
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
    if !opts.lora_specs.is_empty() {
        let loras = load_lora_set(&opts.lora_specs)?;
        runner.bind_lora(&loras)?;
    }
    let dev = runner.device_name();
    let rep = runner.profile(n_runs, 512)?;
    print_metal_profile_report(&model_name, &dev, &rep);
    Ok(())
}

fn cmd_profile(opts: RunOpts) -> Result<()> {
    if opts.use_cuda {
        #[cfg(feature = "cuda")]
        return cmd_profile_cuda(opts);
        #[cfg(not(feature = "cuda"))]
        bail!("--cuda requires compiling with feature 'cuda'");
    }
    if opts.use_vulkan {
        #[cfg(feature = "vulkan")]
        return cmd_profile_vulkan(opts);
        #[cfg(not(feature = "vulkan"))]
        bail!("--vulkan requires compiling with feature 'vulkan'");
    }
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

    if arch == "inkling2" || arch == "inkling2_loop" {
        let model = Inkling2Model::from_gguf(&gguf)?;
        let st = &model.stats;
        eprintln!(
            "[tinymodel] inkling2 placement: dense={:.1}MiB | ram_mmap_layers={} | ram_owned_layers={} ({:.1}MiB, core_in_fast={}/12) | ssd_stream_layers={} ({:.1}MiB/tok, uncached={})",
            st.dense_bytes as f64 / 1048576.0,
            st.ram_mmap_expert_layers,
            st.ram_owned_expert_layers,
            st.ram_owned_bytes as f64 / 1048576.0,
            st.core_layers_in_fast_tiers,
            st.nvme_direct_expert_layers,
            st.nvme_bytes_per_decode_tok as f64 / 1048576.0,
            st.direct_io_active,
        );
        let mut state = Inkling2State::new_with_kv_type(&model, opts.max_ctx, opts.kv_type);
        model.forward_token(&mut state, &pool, kt, 1, 0, true)?;
        state.reset();

        let n_pp = 32usize.min(opts.max_ctx / 2);
        let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
        let t_pp = Instant::now();
        model.forward_batch(&mut state, &pool, kt, &pp_ids, 0, true)?;
        let pp_ms = t_pp.elapsed().as_secs_f64() * 1e3;
        let pp_tps = (n_pp as f64) / (pp_ms * 1e-3).max(1e-9);

        let n_runs = opts.n_predict.max(4);
        let mut prof = Inkling2DecodeProfile::default();
        let mut tok = argmax(state.logits());
        let t_dec = Instant::now();
        for step in 0..n_runs {
            model.forward_token_profiled(&mut state, &pool, kt, tok, n_pp + step, &mut prof)?;
            tok = argmax(state.logits());
        }
        let wall_ns = t_dec.elapsed().as_nanos() as u64;
        let runs_f = prof.n_steps.max(1) as f64;
        let wall_ms = (wall_ns as f64 / 1e6) / runs_f;
        let dec_tps = 1e3 / wall_ms.max(1e-6);
        let total_b =
            (prof.dense_attn_bytes + prof.moe_ram_bytes + prof.moe_ssd_bytes + prof.lm_head_bytes)
                as f64
                / runs_f;
        let e2e_gbs = (total_b / 1e9) / (wall_ms * 1e-3);

        println!(
            "=== CPU Profile (Inkling-2 Looped MoE): {} (isa={}, threads={}, kv={}) ===",
            model_name,
            kt.isa.name(),
            pool.num_threads(),
            opts.kv_type.as_str()
        );
        println!(
            "Prefill pp{}: {:.2} ms ({:.2} tok/s) | Decode tg{}: {:.2} ms/tok ({:.2} tok/s, eff_bw={:.2} GB/s)",
            n_pp, pp_ms, pp_tps, n_runs, wall_ms, dec_tps, e2e_gbs
        );
        println!();
        println!(
            "{:<26} {:>10} {:>8} {:>10} {:>12}",
            "Decode Stage", "ms/tok", "Share%", "MiB/tok", "Eff GB/s"
        );
        println!("{}", "-".repeat(70));
        let rows = [
            ("embed+depth_mix", prof.embed_mix_ns, 0u64),
            ("loop_controllers (24v)", prof.loop_ctrl_ns, 0u64),
            (
                "dense_attn_proj+conv",
                prof.dense_attn_proj_ns,
                prof.dense_attn_bytes,
            ),
            ("gqa_rel_sink_kv", prof.gqa_kv_ns, 0u64),
            ("moe_router+post_conv", prof.moe_route_ns, 0u64),
            ("moe_experts_ram", prof.moe_ram_ns, prof.moe_ram_bytes),
            ("moe_experts_ssd", prof.moe_ssd_ns, prof.moe_ssd_bytes),
            ("lm_head (200K BF16)", prof.lm_head_ns, prof.lm_head_bytes),
        ];
        for (name, ns, bytes) in rows {
            let ms = (ns as f64 / 1e6) / runs_f;
            let pct = 100.0 * (ns as f64) / (wall_ns as f64).max(1.0);
            let b_tok = bytes as f64 / runs_f;
            let mib = b_tok / (1024.0 * 1024.0);
            let gbs = if b_tok > 0.0 && ms > 0.0 {
                (b_tok / 1e9) / (ms * 1e-3)
            } else {
                0.0
            };
            println!(
                "{:<26} {:>10.2} {:>7.1}% {:>10.1} {:>12.2}",
                name, ms, pct, mib, gbs
            );
        }
        return Ok(());
    }

    if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
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
    let loras = if !opts.lora_specs.is_empty() {
        Some(load_lora_set(&opts.lora_specs)?)
    } else {
        None
    };

    let file = File::create(&out_path)
        .with_context(|| format!("failed to create '{}'", out_path.display()))?;
    let mut writer = BufWriter::new(file);

    if opts.use_vulkan {
        #[cfg(feature = "vulkan")]
        {
            if arch == "inkling2" || arch == "inkling2_loop" {
                let pool = ThreadPool::new(opts.threads);
                let kt = active_kernels();
                let mut runner =
                    tm_vulkan::Inkling2VkRunner::from_gguf(gguf.clone(), opts.max_ctx)?;
                let mut host_helper = if runner.has_host_offload_layers() {
                    let hm = tm_models::Inkling2Model::from_gguf_with_options(
                        &gguf,
                        &tm_models::Inkling2LoadOptions {
                            vram_expert_layers: runner.vram_expert_mask().to_vec(),
                            ram_expert_budget_bytes: None,
                            host_moe_only: true,
                        },
                    )?;
                    let hs = tm_models::Inkling2State::new_host_moe_only(&hm);
                    Some((hm, hs))
                } else {
                    None
                };
                let vocab = runner.cfg.vocab_size as u32;
                writer.write_all(b"TMLG")?;
                writer.write_all(&1u32.to_le_bytes())?;
                writer.write_all(&(n_steps as u32).to_le_bytes())?;
                writer.write_all(&vocab.to_le_bytes())?;
                for &id in &input_ids {
                    writer.write_all(&id.to_le_bytes())?;
                }
                for (pos, &id) in input_ids.iter().enumerate() {
                    let logits = eval_inkling2_vk_token(
                        &mut runner,
                        &mut host_helper,
                        &pool,
                        kt,
                        id,
                        pos,
                        true,
                    )?;
                    write_f32_le(&mut writer, logits)?;
                }
                writer.flush()?;
                eprintln!(
                    "[tinymodel logits] wrote {} steps x {} vocab (vulkan {} {}) -> {}",
                    n_steps,
                    vocab,
                    arch,
                    runner.weights_mode(),
                    out_path.display()
                );
                return Ok(());
            }
            if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
                let mut runner =
                    tm_vulkan::Qwen35VkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
                if let Some(ref ls) = loras {
                    runner.bind_lora(ls)?;
                }
                let vocab = runner.cfg.vocab_size as u32;
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
                    "[tinymodel logits] wrote {} steps x {} vocab (vulkan {}, loras={}) -> {}",
                    n_steps,
                    vocab,
                    arch,
                    opts.lora_specs.len(),
                    out_path.display()
                );
                return Ok(());
            }
            let mut runner =
                tm_vulkan::LlamaVkRunner::from_gguf_with_kv(gguf, opts.max_ctx, opts.kv_type)?;
            if let Some(ref ls) = loras {
                runner.bind_lora(ls)?;
            }
            let vocab = runner.cfg.vocab_size as u32;
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
                "[tinymodel logits] wrote {} steps x {} vocab (vulkan {}, loras={}) -> {}",
                n_steps,
                vocab,
                arch,
                opts.lora_specs.len(),
                out_path.display()
            );
            return Ok(());
        }
        #[cfg(not(feature = "vulkan"))]
        bail!("--vulkan requires compiling with feature 'vulkan'");
    }

    if opts.use_cuda {
        #[cfg(feature = "cuda")]
        {
            let mut runner =
                tm_cuda::CudaRunner::from_gguf_with_kv_type(&gguf, opts.max_ctx, opts.kv_type)?;
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
                "[tinymodel logits] wrote {} steps x {} vocab (cuda {}) -> {}",
                n_steps,
                vocab,
                arch,
                out_path.display()
            );
            return Ok(());
        }
        #[cfg(not(feature = "cuda"))]
        bail!("--cuda requires compiling with feature 'cuda'");
    }

    if opts.use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        {
            let mut runner = tm_metal::MetalRunner::from_gguf(gguf, opts.max_ctx)?;
            if let Some(ref ls) = loras {
                runner.bind_lora(ls)?;
            }
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
                "[tinymodel logits] wrote {} steps x {} vocab (metal {}, loras={}) -> {}",
                n_steps,
                vocab,
                arch,
                opts.lora_specs.len(),
                out_path.display()
            );
            return Ok(());
        }
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
    }

    let pool = ThreadPool::new(opts.threads);
    let kt = active_kernels();
    let mut model = CpuModel::from_gguf(&gguf)?;
    if let Some(ref ls) = loras {
        model.bind_lora(&gguf, ls)?;
    }
    check_linux_dirty_budget(
        model.estimate_dirty_bytes_with_kv(opts.max_ctx, opts.kv_type),
        opts.max_ctx,
    )?;
    let mut state = model.new_state_with_kv(opts.max_ctx, opts.kv_type);
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

#[cfg(feature = "train")]
fn enforce_thermal_guard(max_temp_c: Option<f64>) {
    let Some(limit) = max_temp_c else {
        return;
    };
    if let Some(mut cur) = probe::collect_telemetry().max_temp_c
        && cur > limit
    {
        let target = limit - 3.0;
        eprintln!(
            "[tinymodel thermal] SoC temp {:.1}°C > {:.1}°C limit; pausing until <= {:.1}°C...",
            cur, limit, target
        );
        while cur > target {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let Some(next) = probe::collect_telemetry().max_temp_c else {
                break;
            };
            cur = next;
        }
    }
}

#[cfg(feature = "train")]
fn cmd_train_mtp(args: &[String]) -> Result<()> {
    let mut model_path: Option<PathBuf> = None;
    let mut data_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut threads = tm_cpu::recommended_threads();
    let mut max_temp_c: Option<f64> = None;
    let mut json = false;
    let mut cfg = tm_train::MtpTrainConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -m/--model")?,
                ));
            }
            "--data" | "-d" => {
                i += 1;
                data_path = Some(PathBuf::from(
                    args.get(i).context("missing value for --data")?,
                ));
            }
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -o/--output")?,
                ));
            }
            "--steps" => {
                i += 1;
                cfg.steps = args
                    .get(i)
                    .context("missing --steps")?
                    .parse()
                    .context("invalid --steps")?;
            }
            "--seq-len" | "-c" => {
                i += 1;
                cfg.seq_len = args
                    .get(i)
                    .context("missing --seq-len")?
                    .parse()
                    .context("invalid --seq-len")?;
            }
            "--lr" => {
                i += 1;
                cfg.lr = args
                    .get(i)
                    .context("missing --lr")?
                    .parse()
                    .context("invalid --lr")?;
            }
            "--warmup" => {
                i += 1;
                cfg.warmup_steps = args
                    .get(i)
                    .context("missing --warmup")?
                    .parse()
                    .context("invalid --warmup")?;
            }
            "-q" | "--quant" => {
                i += 1;
                let raw = args.get(i).context("missing --quant")?;
                cfg.out_dtype = parse_convert_dtype("--quant", raw)?;
            }
            "-s" | "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .context("missing --seed")?
                    .parse()
                    .context("invalid --seed")?;
            }
            "-t" | "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .context("missing -t")?
                    .parse()
                    .context("invalid -t")?;
            }
            "--max-temp-c" => {
                i += 1;
                max_temp_c = Some(
                    args.get(i)
                        .context("missing --max-temp-c")?
                        .parse()
                        .context("invalid --max-temp-c")?,
                );
            }
            "--json" => json = true,
            other => bail!("unknown train-mtp argument: {other}"),
        }
        i += 1;
    }

    let mp = model_path.context("train-mtp requires -m <base.gguf>")?;
    let dp = data_path.context("train-mtp requires --data <path>")?;
    let op = out_path.context("train-mtp requires -o <mtp.gguf>")?;

    let gguf = GgufFile::open(&mp)?;
    let tok = Tokenizer::from_gguf(&gguf).ok();
    let ds = tm_train::TrainDataset::load(&dp, tok.as_ref(), cfg.seq_len)?;
    let model = CpuModel::from_gguf(&gguf)?;
    let pool = ThreadPool::new(threads);

    if !json {
        eprintln!(
            "[train-mtp] model={} arch={} n_embd={} samples={} steps={} seq_len={} lr={} out_dtype={:?}",
            mp.display(),
            model.arch(),
            model.n_embd(),
            ds.samples.len(),
            cfg.steps,
            cfg.seq_len,
            cfg.lr,
            cfg.out_dtype
        );
    }

    let log_every = cfg.log_every.max(1);
    let (head, rep) =
        tm_train::train_mtp_head(&model, &pool, &ds, &cfg, |step, loss, acc1, lr| {
            enforce_thermal_guard(max_temp_c);
            if !json && (step == 1 || step.is_multiple_of(log_every) || step == cfg.steps) {
                eprintln!(
                    "  step {:4}/{} | loss={:.4} | acc@1={:5.1}% | lr={:.2e}",
                    step,
                    cfg.steps,
                    loss,
                    acc1 * 100.0,
                    lr
                );
            }
        })?;

    let file_bytes = head.save_gguf(&op, model.arch(), cfg.out_dtype)?;
    let mstats = probe_process_memory_stats();
    let tps = (rep.total_mtp_tokens as f64) / rep.elapsed_sec.max(1e-9);
    if json {
        println!(
            "{{\"mode\":\"train-mtp\",\"arch\":\"{}\",\"steps\":{},\"mtp_tokens\":{},\
            \"initial_loss\":{:.4},\"final_loss\":{:.4},\"initial_acc1\":{:.4},\"final_acc1\":{:.4},\
            \"elapsed_sec\":{:.3},\"tok_per_sec\":{:.2},\"out_bytes\":{},\
            \"rss_bytes\":{},\"peak_rss_bytes\":{},\"private_dirty_bytes\":{}}}",
            probe::json_escape(model.arch()),
            cfg.steps,
            rep.total_mtp_tokens,
            rep.initial_loss,
            rep.final_loss,
            rep.initial_acc1,
            rep.final_acc1,
            rep.elapsed_sec,
            tps,
            file_bytes,
            probe::opt_u64_json(mstats.rss_bytes),
            probe::opt_u64_json(mstats.peak_rss_bytes),
            probe::opt_u64_json(mstats.private_dirty_bytes),
        );
    } else {
        println!(
            "train-mtp: saved {} ({:.2} MiB) | loss {:.4} -> {:.4} | acc@1 {:.1}% -> {:.1}% | {:.1} tok/s ({:.2}s)",
            op.display(),
            file_bytes as f64 / 1048576.0,
            rep.initial_loss,
            rep.final_loss,
            rep.initial_acc1 * 100.0,
            rep.final_acc1 * 100.0,
            tps,
            rep.elapsed_sec,
        );
    }
    Ok(())
}

#[cfg(all(feature = "train", feature = "metal", target_vendor = "apple"))]
struct CliMetalAccel {
    inner: tm_metal::MetalGpuAccel,
    dev_name: String,
}

#[cfg(all(feature = "train", feature = "metal", target_vendor = "apple"))]
impl tm_train::LinearAccel for CliMetalAccel {
    fn backend_name(&self) -> &str {
        "metal"
    }

    fn device_name(&self) -> &str {
        &self.dev_name
    }

    fn matmul_fwd(
        &mut self,
        name: &str,
        _dtype: DType,
        _w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        n_tok: usize,
    ) -> Result<bool, String> {
        Ok(self.inner.matmul_fwd(name, rows, cols, x, out, n_tok))
    }

    fn matmul_bwd_transposed(
        &mut self,
        name: &str,
        _dtype: DType,
        _w_bytes: &[u8],
        dy: &[f32],
        dx: &mut [f32],
        rows: usize,
        cols: usize,
        n_tok: usize,
        accum: bool,
    ) -> Result<bool, String> {
        Ok(self
            .inner
            .matmul_bwd_transposed(name, rows, cols, dy, dx, n_tok, accum))
    }
}

#[cfg(all(feature = "train", feature = "vulkan"))]
struct CliVulkanAccel {
    inner: tm_vulkan::VulkanGpuAccel,
    dev_name: String,
}

#[cfg(all(feature = "train", feature = "vulkan"))]
impl tm_train::LinearAccel for CliVulkanAccel {
    fn backend_name(&self) -> &str {
        "vulkan"
    }

    fn device_name(&self) -> &str {
        &self.dev_name
    }

    fn matmul_fwd(
        &mut self,
        name: &str,
        _dtype: DType,
        _w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        n_tok: usize,
    ) -> Result<bool, String> {
        Ok(self.inner.matmul_fwd(name, rows, cols, x, out, n_tok))
    }

    fn matmul_bwd_transposed(
        &mut self,
        name: &str,
        _dtype: DType,
        _w_bytes: &[u8],
        dy: &[f32],
        dx: &mut [f32],
        rows: usize,
        cols: usize,
        n_tok: usize,
        accum: bool,
    ) -> Result<bool, String> {
        Ok(self
            .inner
            .matmul_bwd_transposed(name, rows, cols, dy, dx, n_tok, accum))
    }
}

#[cfg(feature = "train")]
fn cmd_finetune(args: &[String]) -> Result<()> {
    let mut model_path: Option<PathBuf> = None;
    let mut data_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut threads = tm_cpu::recommended_threads();
    let mut max_temp_c: Option<f64> = None;
    let mut json = false;
    let mut log_every = 5usize;
    let mut cfg = tm_train::QloraConfig::default();
    let mut alpha_explicit = false;
    let mut use_metal = false;
    let mut use_vulkan = false;
    let mut use_cuda = false;
    let mut gpu_min_macs: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -m/--model")?,
                ));
            }
            "--data" | "-d" => {
                i += 1;
                data_path = Some(PathBuf::from(
                    args.get(i).context("missing value for --data")?,
                ));
            }
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i).context("missing value for -o/--output")?,
                ));
            }
            "--rank" | "-r" => {
                i += 1;
                cfg.rank = args
                    .get(i)
                    .context("missing --rank")?
                    .parse()
                    .context("invalid --rank")?;
            }
            "--alpha" => {
                i += 1;
                cfg.alpha = args
                    .get(i)
                    .context("missing --alpha")?
                    .parse()
                    .context("invalid --alpha")?;
                alpha_explicit = true;
            }
            "--targets" => {
                i += 1;
                let raw = args.get(i).context("missing --targets")?;
                cfg.targets = tm_train::LoraTargets::parse_csv(raw)?;
            }
            "--steps" => {
                i += 1;
                cfg.steps = args
                    .get(i)
                    .context("missing --steps")?
                    .parse()
                    .context("invalid --steps")?;
            }
            "--seq-len" | "-c" => {
                i += 1;
                cfg.seq_len = args
                    .get(i)
                    .context("missing --seq-len")?
                    .parse()
                    .context("invalid --seq-len")?;
            }
            "--grad-accum" => {
                i += 1;
                cfg.grad_accum = args
                    .get(i)
                    .context("missing --grad-accum")?
                    .parse()
                    .context("invalid --grad-accum")?;
            }
            "--lr" => {
                i += 1;
                cfg.optim.lr = args
                    .get(i)
                    .context("missing --lr")?
                    .parse()
                    .context("invalid --lr")?;
            }
            "--lora-plus-ratio" => {
                i += 1;
                cfg.optim.lora_plus_ratio = args
                    .get(i)
                    .context("missing --lora-plus-ratio")?
                    .parse()
                    .context("invalid --lora-plus-ratio")?;
            }
            "--weight-decay" => {
                i += 1;
                cfg.optim.weight_decay = args
                    .get(i)
                    .context("missing --weight-decay")?
                    .parse()
                    .context("invalid --weight-decay")?;
            }
            "--max-grad-norm" => {
                i += 1;
                cfg.optim.max_grad_norm = args
                    .get(i)
                    .context("missing --max-grad-norm")?
                    .parse()
                    .context("invalid --max-grad-norm")?;
            }
            "--warmup" => {
                i += 1;
                cfg.warmup_steps = args
                    .get(i)
                    .context("missing --warmup")?
                    .parse()
                    .context("invalid --warmup")?;
            }
            "--save-every" => {
                i += 1;
                cfg.save_every = args
                    .get(i)
                    .context("missing --save-every")?
                    .parse()
                    .context("invalid --save-every")?;
            }
            "--ckpt" => {
                i += 1;
                cfg.ckpt_path = Some(PathBuf::from(args.get(i).context("missing --ckpt")?));
            }
            "--resume" => {
                i += 1;
                cfg.resume_ckpt = Some(PathBuf::from(args.get(i).context("missing --resume")?));
            }
            "--f16" => cfg.save_f16_adapter = true,
            "--fp32-ckpt" => cfg.checkpoint_mode = tm_train::CheckpointMode::Fp32,
            "--log-every" => {
                i += 1;
                log_every = args
                    .get(i)
                    .context("missing --log-every")?
                    .parse()
                    .context("invalid --log-every")?;
            }
            "-s" | "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .context("missing --seed")?
                    .parse()
                    .context("invalid --seed")?;
            }
            "-t" | "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .context("missing -t")?
                    .parse()
                    .context("invalid -t")?;
            }
            "--max-temp-c" => {
                i += 1;
                max_temp_c = Some(
                    args.get(i)
                        .context("missing --max-temp-c")?
                        .parse()
                        .context("invalid --max-temp-c")?,
                );
            }
            "--metal" => {
                use_metal = true;
                use_vulkan = false;
                use_cuda = false;
            }
            "--vulkan" => {
                use_vulkan = true;
                use_metal = false;
                use_cuda = false;
            }
            "--cuda" => {
                use_cuda = true;
                use_metal = false;
                use_vulkan = false;
            }
            "--gpu" => {
                i += 1;
                let bk = args.get(i).context("missing value for --gpu")?;
                match bk.as_str() {
                    "cuda" | "nv" => {
                        use_cuda = true;
                        use_vulkan = false;
                        use_metal = false;
                    }
                    "vulkan" | "vk" => {
                        use_vulkan = true;
                        use_metal = false;
                        use_cuda = false;
                    }
                    "metal" | "mtl" => {
                        use_metal = true;
                        use_vulkan = false;
                        use_cuda = false;
                    }
                    "cpu" | "none" => {
                        use_vulkan = false;
                        use_metal = false;
                        use_cuda = false;
                    }
                    other => {
                        bail!("unknown --gpu backend '{other}' (expected cuda, vulkan, metal, cpu)")
                    }
                }
            }
            "--gpu-min-macs" => {
                i += 1;
                gpu_min_macs = Some(
                    args.get(i)
                        .context("missing --gpu-min-macs")?
                        .parse()
                        .context("invalid --gpu-min-macs")?,
                );
            }
            "--json" => json = true,
            other => bail!("unknown finetune argument: {other}"),
        }
        i += 1;
    }

    if use_cuda {
        bail!("finetune --cuda is not yet wired (use --metal, --vulkan, or CPU)");
    }
    if use_metal && use_vulkan {
        bail!("cannot specify both --metal and --vulkan");
    }
    if !alpha_explicit {
        cfg.alpha = (2 * cfg.rank) as f32;
    }

    let mp = model_path.context("finetune requires -m <base.gguf>")?;
    let dp = data_path.context("finetune requires --data <path>")?;
    let op = out_path.context("finetune requires -o <adapter.gguf>")?;

    let tel_before = probe::collect_telemetry();
    let gguf = std::sync::Arc::new(GgufFile::open(&mp)?);
    let arch = gguf.req_str("general.architecture")?.to_string();
    let tok = Tokenizer::from_gguf(&gguf).ok();
    // Load up to `seq_len + 1` tokens per sample (`seq_len` input tokens + 1 shifted label)
    let ds = tm_train::TrainDataset::load(&dp, tok.as_ref(), cfg.seq_len + 1)?;
    let model = CpuModel::from_gguf(&gguf)?;
    let pool = ThreadPool::new(threads);
    let kt = active_kernels();

    let mut accel: Option<Box<dyn tm_train::LinearAccel>> = None;
    if use_metal {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        {
            let mut ma = tm_metal::MetalGpuAccel::from_gguf(gguf.clone(), cfg.seq_len.max(2))?;
            if let Some(mm) = gpu_min_macs {
                ma = ma.with_min_macs(mm);
            }
            let dev_name = ma.device_name();
            accel = Some(Box::new(CliMetalAccel {
                inner: ma,
                dev_name,
            }));
        }
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        {
            let _ = gpu_min_macs;
            bail!("--metal requires compiling with feature 'metal' on Apple Silicon");
        }
    } else if use_vulkan {
        #[cfg(feature = "vulkan")]
        {
            let mut va = tm_vulkan::VulkanGpuAccel::from_gguf(gguf.clone(), cfg.seq_len.max(2))?;
            if let Some(mm) = gpu_min_macs {
                va = va.with_min_macs(mm);
            }
            let dev_name = va.device_name();
            accel = Some(Box::new(CliVulkanAccel {
                inner: va,
                dev_name,
            }));
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = gpu_min_macs;
            bail!("--vulkan requires compiling with feature 'vulkan'");
        }
    } else {
        let _ = gpu_min_macs;
    }

    let backend_label = accel
        .as_ref()
        .map(|a| a.backend_name().to_string())
        .unwrap_or_else(|| "cpu".to_string());
    let device_label = accel
        .as_ref()
        .map(|a| a.device_name().to_string())
        .unwrap_or_else(probe::detect_device_model_name);

    if !json {
        eprintln!(
            "[finetune] model={} arch={} backend={} ({}) isa={} threads={} rank={} alpha={} steps={} seq_len={} grad_accum={} lr={} lora+={}",
            mp.display(),
            arch,
            backend_label,
            device_label,
            kt.isa.name(),
            pool.num_threads(),
            cfg.rank,
            cfg.alpha,
            cfg.steps,
            cfg.seq_len,
            cfg.grad_accum,
            cfg.optim.lr,
            cfg.optim.lora_plus_ratio,
        );
    }

    let le = log_every.max(1);
    let (adapter, rep) = tm_train::run_cpu_qlora_training(
        &model,
        &pool,
        &ds,
        &cfg,
        accel.as_deref_mut(),
        |m| {
            enforce_thermal_guard(max_temp_c);
            if !json && (m.step == 1 || m.step.is_multiple_of(le) || m.step == cfg.steps) {
                eprintln!(
                    "  step {:4}/{} | loss={:.4} | grad_norm={:.4} | lr={:.2e} | sup_tok={}/{} | {:.1} ms ({:.1} tok/s)",
                    m.step,
                    cfg.steps,
                    m.loss,
                    m.grad_norm,
                    m.lr,
                    m.supervised_tokens,
                    m.total_tokens,
                    m.step_ms,
                    (m.total_tokens as f64) / (m.step_ms * 1e-3).max(1e-9),
                );
            }
            Ok(())
        },
    )?;

    let adapter_bytes = adapter.save_gguf(&op, cfg.save_f16_adapter)?;
    let tel_after = probe::collect_telemetry();
    let mstats = probe_process_memory_stats();
    let tok_s = (rep.total_tokens as f64) / rep.elapsed_sec.max(1e-9);
    if json {
        println!(
            "{{\"mode\":\"finetune\",\"model\":\"{}\",\"arch\":\"{}\",\"backend\":\"{}\",\"device_name\":\"{}\",\"isa\":\"{}\",\"threads\":{},\
            \"rank\":{},\"alpha\":{},\"trainable_params\":{},\"steps\":{},\"seq_len\":{},\"grad_accum\":{},\
            \"total_tokens\":{},\"supervised_tokens\":{},\"initial_loss\":{:.5},\"final_loss\":{:.5},\"min_loss\":{:.5},\
            \"elapsed_sec\":{:.3},\"tok_per_sec\":{:.2},\"adapter_bytes\":{},\"est_dirty_bytes\":{},\
            \"rss_bytes\":{},\"peak_rss_bytes\":{},\"private_dirty_bytes\":{},\"pss_anon_bytes\":{},\
            \"telemetry_before\":{},\"telemetry_after\":{}}}",
            probe::json_escape(mp.file_name().and_then(|s| s.to_str()).unwrap_or("?")),
            probe::json_escape(&arch),
            probe::json_escape(&backend_label),
            probe::json_escape(&device_label),
            probe::json_escape(kt.isa.name()),
            pool.num_threads(),
            cfg.rank,
            cfg.alpha,
            rep.trainable_params,
            rep.total_steps,
            cfg.seq_len,
            cfg.grad_accum,
            rep.total_tokens,
            rep.supervised_tokens,
            rep.initial_loss,
            rep.final_loss,
            rep.min_loss,
            rep.elapsed_sec,
            tok_s,
            adapter_bytes,
            rep.estimated_dirty_bytes,
            probe::opt_u64_json(mstats.rss_bytes),
            probe::opt_u64_json(mstats.peak_rss_bytes),
            probe::opt_u64_json(mstats.private_dirty_bytes),
            probe::opt_u64_json(mstats.pss_anon_bytes),
            tel_before.to_json(),
            tel_after.to_json(),
        );
    } else {
        println!(
            "finetune: saved {} ({:.2} MiB, {} trainable params, backend={}) | loss {:.4} -> {:.4} (min {:.4}) | {:.1} tok/s ({:.2}s)",
            op.display(),
            adapter_bytes as f64 / 1048576.0,
            rep.trainable_params,
            backend_label,
            rep.initial_loss,
            rep.final_loss,
            rep.min_loss,
            tok_s,
            rep.elapsed_sec,
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
        "probe" => probe::cmd_probe(probe::parse_probe_opts(&args[2..])?),
        "inspect" => {
            let p = args.get(2).context("missing <model.gguf>")?;
            cmd_inspect(PathBuf::from(p))
        }
        "convert" => cmd_convert(&args[2..]),
        "lora-merge" => cmd_lora_merge(&args[2..]),
        "run" => cmd_run(parse_run_opts(&args[2..])?),
        "bench" => cmd_bench(parse_run_opts(&args[2..])?),
        "profile" => cmd_profile(parse_run_opts(&args[2..])?),
        "logits" => cmd_logits(parse_run_opts(&args[2..])?),
        #[cfg(feature = "train")]
        "train-mtp" => cmd_train_mtp(&args[2..]),
        #[cfg(not(feature = "train"))]
        "train-mtp" => bail!("'train-mtp' requires compiling with feature 'train'"),
        #[cfg(feature = "train")]
        "finetune" => cmd_finetune(&args[2..]),
        #[cfg(not(feature = "train"))]
        "finetune" => bail!("'finetune' requires compiling with feature 'train'"),
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
