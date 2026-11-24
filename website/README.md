# tinymodel Website

`tinymodel` 项目官网，采用 **纯静态、零构建依赖（HTML + CSS + 原生 ES Module JS）** 实现，契合项目自身“零膨胀、极致轻量”的工程哲学。

- **无 `node_modules` / 无构建步骤**：任意静态托管平台直接指定 `website/` 目录即可上线。
- **超低/零成本**：支持 Cloudflare Pages、GitHub Pages、Vercel、Netlify 等免费全球 CDN，以及对象存储（OSS/COS/S3）或 ~12MB Alpine Nginx 容器。
- **特性**：中英双语切换、深色/浅色主题、交互式硬件三档矩阵、真实实测基准对比图表、交互式 CLI 命令生成器、项目路线图。

---

## 本地预览

无需安装任何依赖，在仓库根目录运行任意静态文件服务即可：

```sh
python3 -m http.server 8000 --directory website
# 打开 http://localhost:8000
```

---

## 低成本云平台部署指南

### 1. Cloudflare Pages（最推荐：$0 / 无限流量 / 全球 CDN）

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**。
2. 选择本仓库，按如下配置：
   - **Framework preset**: `None`
   - **Build command**: 留空
   - **Build output directory**: `website`
3. 点击 **Save and Deploy**。仓库内已内置 `website/_headers`（安全头与静态资源缓存策略）。

或使用 CLI 一键部署：

```sh
npx wrangler pages deploy website --project-name tinymodel
```

### 2. GitHub Pages（$0 / 自动化 CI）

仓库已包含 `.github/workflows/deploy-website.yml`：
1. 在 GitHub 仓库进入 **Settings → Pages**。
2. **Source** 选择 **GitHub Actions**。
3. 合并到 `main` 分支（或手动触发 `workflow_dispatch`）即可自动部署上线。

### 3. Vercel（$0 Hobby 档）

1. 导入仓库后，将 **Root Directory** 设为 `website`。
2. **Framework Preset** 选择 `Other`，**Build Command** 留空。
3. 自动应用 `website/vercel.json` 的缓存与安全头。

或使用 CLI：

```sh
npx vercel --cwd website --prod
```

### 4. Netlify（$0 Starter 档）

1. 导入仓库后，将 **Base directory** 设为 `website`，**Publish directory** 设为 `.`，**Build command** 留空。
2. 自动应用 `website/netlify.toml`。

### 5. 国内云对象存储 + CDN（极低成本，月费约 ¥0–¥5）

适合阿里云 OSS、腾讯云 COS、华为云 OBS 或七牛云：
1. 开启 Bucket 的 **静态网站托管**，默认首页设为 `index.html`。
2. 将 `website/` 下的 `index.html`、`assets/`、`css/`、`js/` 上传至 Bucket 根目录。
3. 绑定自定义域名并开启 HTTPS / CDN 缓存。

### 6. 极简容器部署（Fly.io / Cloud Run / 自建轻量服务器）

镜像基于 `nginx:1.27-alpine`，体积约 **12 MB**，内存占用 < 10 MB，支持 gzip 与静态资源长缓存：

```sh
docker build -t tinymodel-web website/
docker run -d -p 8080:8080 tinymodel-web
```
