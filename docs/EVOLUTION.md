# Quant-AI Evolution Roadmap

> 当前状态：V2+V3 Batch 1-10 已完成，CI ✅，Render 轻量部署 ✅
> 公网 URL：https://quant-ai-qzrg.onrender.com

---

## 🔴 Phase 1 — 核心修复（立刻做）

### 1.1 ✅ CI 修复
- [x] F541 lint 忽略
- [x] E402 lint 忽略（conditional imports）
- [x] 用 requirements.txt 安装依赖
- [x] 契约测试加入 CI

### 1.2 ✅ Render 轻量部署
- [x] RAG/FAISS 改为可选（lazy import + fallback）
- [x] sentence-transformers 移到 requirements-full.txt
- [x] Render Free plan 部署成功（~200MB）

### 1.3 跑真实数据验证（拿简历 metrics）
- [ ] 通过 /train API 训练 2-3 个模型（AAPL + MSFT + GOOGL）
- [ ] 记录实际 AUC / Accuracy / F1
- [ ] 通过 /backtest API 跑回测
- [ ] 记录实际 Sharpe / CAGR / MaxDD
- [ ] 截图 Swagger UI 放 README

---

## 🟡 Phase 2 — RAG + 云平台升级（AWS 激活后）

### 2.1 AWS 部署（App Runner）
- [ ] 完成 AWS 账号激活
- [ ] 创建 App Runner service（1 vCPU + 2GB RAM）
- [ ] 配置 GitHub 自动部署
- [ ] 设置 Supabase 环境变量
- [ ] 迁移公网 URL

### 2.2 恢复 RAG 功能
- [ ] 在 AWS 上安装 requirements-full.txt（含 sentence-transformers + faiss-cpu）
- [ ] 验证 /search 和 /rag/answer 端点
- [ ] FAISS 索引持久化（S3 或本地磁盘）

### 2.3 Supabase 连通
- [ ] 配置 SUPABASE_URL + SUPABASE_KEY 环境变量
- [ ] model_registry 真正写入 Supabase
- [ ] training_runs 记录写入 Supabase
- [ ] 启用 RLS（Row Level Security）

---

## 🟢 Phase 3 — 面试级增强

### 3.1 UI 可视化
- [ ] Dashboard 加价格走势图（Recharts / Chart.js）
- [ ] Training 页面加训练进度 + metrics 展示
- [ ] Backtest 页面加收益曲线对比图
- [ ] 部署前端（Vercel / Render Static）

### 3.2 测试 + 质量
- [ ] 契约测试全部通过（不用 || true）
- [ ] 添加 coverage badge 到 README
- [ ] 添加 E2E 测试（训练→预测→回测完整流程）

### 3.3 Observability
- [ ] Sentry 错误追踪集成
- [ ] 简单 metrics（请求量/延迟/错误率）
- [ ] 结构化日志导出（CloudWatch / Datadog）

### 3.4 Docker Compose 一键跑
- [ ] 验证 docker-compose up 全栈可用（API + Worker + DB）
- [ ] 写 docs/local-setup.md 新手指南

### 3.5 README 增强
- [ ] 添加公网 API URL
- [ ] 添加 Swagger UI 截图
- [ ] 添加部署说明（Render + AWS）
- [ ] 添加真实 backtest 结果截图
- [ ] 添加 CI badge

---

## 📋 修复优先级排序

| 优先级 | 任务 | 预估时间 | 影响 |
|--------|------|---------|------|
| P0 | 1.3 跑真实数据拿 metrics | 30 min | 简历核心数据 |
| P1 | 2.1 AWS 部署 | 1-2h | 生产级部署 + RAG |
| P1 | 2.2 恢复 RAG | 30 min | Full-stack 展示 |
| P1 | 2.3 Supabase 连通 | 1h | 数据持久化 |
| P2 | 3.1 UI 可视化增强 | 2-4h | 面试 demo 效果 |
| P2 | 3.5 README 增强 | 1h | 第一印象 |
| P3 | 3.2 测试覆盖 | 1-2h | 工程规范 |
| P3 | 3.3 Observability | 1-2h | 生产级运维 |
| P3 | 3.4 Docker Compose | 30 min | 开发体验 |
