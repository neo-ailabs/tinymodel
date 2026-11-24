import { I18N, PROFILES, BENCHMARKS, MODELS, ARCH_HIGHLIGHTS, ROADMAP } from "./data.js";

const state = {
  lang: localStorage.getItem("tm_lang") || "zh",
  theme: localStorage.getItem("tm_theme") || "dark",
  benchTab: "vulkan"
};

function t(key) {
  return (I18N[state.lang] && I18N[state.lang][key]) || key;
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = state.theme === "dark" ? "☀ Light" : "☾ Dark";
  }
}

function applyLang() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (key) el.textContent = t(key);
  });
  const langBtn = document.getElementById("lang-toggle");
  if (langBtn) {
    langBtn.textContent = state.lang === "zh" ? "EN" : "中文";
  }
  renderProfiles();
  renderBenchTabs();
  renderBenchmark();
  renderModels();
  renderArchHighlights();
  renderRoadmap();
  updateCliCommand();
}

function renderProfiles() {
  const container = document.getElementById("profiles-grid");
  if (!container) return;
  container.innerHTML = PROFILES.map((p) => {
    const loc = p[state.lang];
    const specsHtml = loc.specs
      .map(([k, v]) => `<li><span>${k}</span><strong>${v}</strong></li>`)
      .join("");
    const devicesHtml = loc.devices
      .map((d, i) => `<span class="chip ${i === 0 ? "green" : ""}">${d}</span>`)
      .join("");
    return `
      <article class="profile-card ${p.featured ? "featured" : ""}">
        <div class="profile-top">
          <span class="profile-badge">${p.badge}</span>
          <span class="profile-scale">${p.scale}</span>
        </div>
        <h3 style="margin:0;font-size:1.1rem;">${loc.title}</h3>
        <ul class="profile-specs">${specsHtml}</ul>
        <div class="profile-devices">${devicesHtml}</div>
        <div class="profile-cmd" title="${p.cmd}">$ ${p.cmd}</div>
      </article>
    `;
  }).join("");
}

function renderBenchTabs() {
  const tabsEl = document.getElementById("bench-tabs");
  if (!tabsEl) return;
  const keys = Object.keys(BENCHMARKS);
  tabsEl.innerHTML = keys
    .map((key) => {
      const active = key === state.benchTab ? "active" : "";
      const label = BENCHMARKS[key][state.lang].label;
      return `<button class="tab-btn ${active}" data-bench="${key}" type="button">${label}</button>`;
    })
    .join("");
  tabsEl.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.benchTab = btn.getAttribute("data-bench");
      renderBenchTabs();
      renderBenchmark();
    });
  });
}

function renderBenchmark() {
  const dataset = BENCHMARKS[state.benchTab];
  if (!dataset) return;
  const loc = dataset[state.lang];
  const noteEl = document.getElementById("bench-note");
  const baseLegendEl = document.getElementById("bench-base-legend");
  const listEl = document.getElementById("bench-list");

  if (noteEl) noteEl.textContent = loc.note;
  if (baseLegendEl) baseLegendEl.textContent = loc.baseName;
  if (!listEl) return;

  listEl.innerHTML = dataset.rows
    .map((row) => {
      const maxVal = Math.max(row.tm, row.base) * 1.06;
      const tmPct = Math.max(8, Math.round((row.tm / maxVal) * 100));
      const basePct = Math.max(8, Math.round((row.base / maxVal) * 100));
      return `
        <div class="bench-row">
          <div class="bench-row-head">
            <div>
              <span class="bench-model">${row.model}</span>
              <span class="bench-sub">${row.sub}</span>
            </div>
            <span class="bench-ratio">${row.ratio}</span>
          </div>
          <div class="bench-bars">
            <div class="bar-line">
              <span class="bar-label">tinymodel</span>
              <div class="bar-track"><div class="bar-fill tm" style="width:${tmPct}%"></div></div>
              <span class="bar-val">${row.tm} tok/s</span>
            </div>
            <div class="bar-line">
              <span class="bar-label">${row.baseLabel}</span>
              <div class="bar-track"><div class="bar-fill base" style="width:${basePct}%"></div></div>
              <span class="bar-val">${row.base} tok/s</span>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderModels() {
  const grid = document.getElementById("models-grid");
  if (!grid) return;
  grid.innerHTML = MODELS.map((m) => {
    const loc = m[state.lang];
    const tags = loc.tags.map((tg) => `<span class="chip">${tg}</span>`).join("");
    return `
      <article class="feature-card">
        <div class="card-kicker">
          <span class="card-icon">${m.kicker}</span>
          <span class="chip blue">${m.status}</span>
        </div>
        <h3>${loc.title}</h3>
        <p>${loc.desc}</p>
        <div class="card-tags">${tags}</div>
      </article>
    `;
  }).join("");
}

function renderArchHighlights() {
  const el = document.getElementById("arch-highlights");
  if (!el) return;
  el.innerHTML = ARCH_HIGHLIGHTS[state.lang]
    .map(
      (item) => `
      <div class="highlight-item">
        <h4>${item.title}</h4>
        <p>${item.desc}</p>
      </div>
    `
    )
    .join("");
}

function renderRoadmap() {
  const el = document.getElementById("roadmap-grid");
  if (!el) return;
  el.innerHTML = ROADMAP.map((item) => {
    const statusClass =
      item.status === "done"
        ? "status-done"
        : item.status === "wip"
        ? "status-wip"
        : "status-todo";
    const stage = state.lang === "zh" ? item.stage : item.stageEn;
    const badge = state.lang === "zh" ? item.badgeZh : item.badgeEn;
    const desc = state.lang === "zh" ? item.zh : item.en;
    return `
      <article class="roadmap-item">
        <div class="roadmap-top">
          <span class="roadmap-stage">${stage}</span>
          <span class="status-pill ${statusClass}">${badge}</span>
        </div>
        <p>${desc}</p>
      </article>
    `;
  }).join("");
}

function updateCliCommand() {
  const mode = document.getElementById("cli-mode")?.value || "run";
  const backend = document.getElementById("cli-backend")?.value || "metal";
  const model = document.getElementById("cli-model")?.value || "qwen35-0.8b.gguf";
  const opt = document.getElementById("cli-opt")?.value || "default";
  const out = document.getElementById("cli-output");
  if (!out) return;

  const gpuFlag = backend === "cpu" ? "" : ` --${backend}`;
  let buildLine = "cargo build --release";
  if (backend === "metal") buildLine = "cargo build --release --features metal";
  if (backend === "vulkan") buildLine = "cargo build --release --features vulkan";
  if (mode === "finetune") buildLine += backend === "cpu" ? " --features train" : ",train";

  let runLine = "";
  if (mode === "run") {
    const optFlags =
      opt === "spec"
        ? " --spec auto --kv-type q8_0"
        : opt === "lora"
        ? " --lora adapter.gguf:1.0"
        : opt === "q8kv"
        ? " --kv-type q8_0"
        : "";
    runLine = `./target/release/tinymodel run -m models/${model} -p "Explain Rust zero-cost abstractions:" -n 128${gpuFlag}${optFlags}`;
  } else if (mode === "bench") {
    const optFlags = opt === "spec" ? " --spec auto" : opt === "q8kv" ? " --kv-type q8_0" : "";
    runLine = `./target/release/tinymodel bench -m models/${model} -n 128${gpuFlag}${optFlags} --json`;
  } else if (mode === "convert") {
    const quantFlag = opt === "micro" ? "--quant q2_0 --prune-vocab 16384" : "--quant q4_0x4 --lm-head-quant q6_k";
    runLine = `./target/release/tinymodel convert -i models/${model} -o models/packed-${model} ${quantFlag}`;
  } else if (mode === "finetune") {
    runLine = `./target/release/tinymodel finetune -m models/${model} --data train.jsonl -o lora.gguf --rank 8 --alpha 16 --steps 100${gpuFlag}`;
  } else if (mode === "probe") {
    runLine = `./target/release/tinymodel probe${backend === "vulkan" ? " --vulkan" : ""} --json`;
  }

  const c1 = state.lang === "zh" ? "# 1. 使用 Rust stable 编译" : "# 1. Build with Rust stable";
  const c2 = state.lang === "zh" ? "# 2. 执行命令" : "# 2. Run command";

  out.innerHTML = `<span class="comment">${c1}</span>\n${buildLine}\n\n<span class="comment">${c2}</span>\n${runLine}`;
  out.dataset.raw = `${buildLine}\n${runLine}`;
}

let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 1800);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast(t("copied"));
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    showToast(t("copied"));
  }
}

function init() {
  applyTheme();
  applyLang();

  document.getElementById("theme-toggle")?.addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    localStorage.setItem("tm_theme", state.theme);
    applyTheme();
  });

  document.getElementById("lang-toggle")?.addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    localStorage.setItem("tm_lang", state.lang);
    applyLang();
  });

  document.getElementById("menu-toggle")?.addEventListener("click", () => {
    document.getElementById("nav-links")?.classList.toggle("open");
  });

  document.querySelectorAll("#nav-links a").forEach((a) => {
    a.addEventListener("click", () => {
      document.getElementById("nav-links")?.classList.remove("open");
    });
  });

  document.getElementById("copy-hero-cmd")?.addEventListener("click", () => {
    copyText("git clone https://github.com/neo-ailabs/tinymodel && cd tinymodel && cargo build --release");
  });

  document.getElementById("copy-cli-cmd")?.addEventListener("click", () => {
    const raw = document.getElementById("cli-output")?.dataset.raw || "";
    if (raw) copyText(raw);
  });

  ["cli-mode", "cli-backend", "cli-model", "cli-opt"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", updateCliCommand);
  });
}

document.addEventListener("DOMContentLoaded", init);
