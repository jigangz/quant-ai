# quant-ai Roadmap

> 目标：从新闻情绪分析工具，成长为完整的 AI 驱动量化研究平台

---

## ✅ 已完成

- [x] 后端基础架构（FastAPI + SQLite/PostgreSQL + Redis）
- [x] Polygon.io 行情 + 新闻数据拉取
- [x] 新闻情绪分析（可选 Claude Haiku 打分）
- [x] 新闻情绪特征工程（7个特征进 ML registry）
- [x] XGBoost 方向预测（T+1 / T+3 / T+5）
- [x] SHAP 特征重要性解释
- [x] 回测引擎（Sharpe / MaxDrawdown / 胜率）
- [x] 模型版本管理（candidate → production 晋升）
- [x] React 前端 Dashboard（D3.js K线图 + 新闻点）
- [x] 前端4页面：Dashboard / Prediction / Backtest / Models

---

## ✅ 第一阶段：夯实基础体验（已完成 2026-03-18）

### 1. 技术指标叠加
- [x] 替换 D3.js → **Lightweight Charts**（TradingView 同款，更流畅）
- [x] 接入 `ta` 库（SMA / EMA / RSI / MACD / 布林带）
- [x] 前端支持叠加显示：MACD / RSI / 布林带 / SMA20 / SMA50
- [x] RSI / MACD 独立子图，时间轴同步
- [x] 多时间周期切换：1D / 1W / 1M / 3M / 1Y

### 2. Python 兼容性修复
- [x] `str | None` 语法改为 `Optional[str]`（兼容 Python 3.9）
- [x] Supabase PostgreSQL 连接配置（替换本地 PostgreSQL 依赖）
- [x] 前端无后端时显示 mock 数据（不白板）+ 橙色提示条

### 3. Screener 选股
- [x] 按涨跌幅 / 成交量排序
- [x] 实时拉取 10 只股票（AAPL/TSLA/NVDA/MSFT 等）
- [x] 点击跳转 Dashboard 并自动加载 K 线
- [x] 前端第5页面：Screener

---

## ✅ 第二阶段：策略系统（已完成 2026-03-19）

### 4. 策略编写系统
- [x] 定义策略基类 `BaseStrategy` + `StrategyRegistry`
- [x] 内置策略模板：
  - 均线交叉（MA Cross）
  - RSI 超买超卖
  - 布林带突破
  - 新闻情绪驱动买卖（quant-ai 独有）
- [x] 策略 + 回测引擎打通，一键回测任意策略
- [x] 前端策略编辑器页面（策略选择 + 参数配置 + 信号可视化 + 回测结果）
- [x] REST API: GET /api/strategies, GET /{name}, POST /{name}/signals, POST /{name}/backtest
- [x] 48 个单元测试全通过

### 5. Paper Trading 模拟盘
- [x] 本地撮合引擎（市价单 / 限价单，支持自动撮合）
- [x] 持仓管理：买入/卖出/持仓记录/平均成本
- [x] 盈亏实时计算（未实现盈亏 + 已实现盈亏）
- [x] WebSocket 实时价格推送
- [x] 前端第6页面：Trading（持仓表 + 下单 + 实时报价 + 权益曲线）
- [x] REST API: 下单/查单/取消/持仓/历史/重置
- [x] Portfolio 快照历史追踪

---

## ✅ 第 2.5 阶段：云原生基础设施（已完成 2026-04-02）

> 引入消息队列、事件驱动计算，提升系统可靠性和可扩展性。
> 所有服务通过抽象接口访问，env var 切换后端，业务代码零改动。

### 部署架构

```
┌─────────────────────────┐    ┌──────────────────────────────┐
│      Render (免费)       │    │     外接服务 (免费层)          │
│                         │    │                              │
│  ├── API Service        │───→│  ├── Upstash Kafka           │
│  │   (FastAPI)          │    │  │   (10,000 msg/天免费)      │
│  │                      │    │  │                            │
│  ├── Worker Service     │───→│  ├── AWS SQS                 │
│  │   (训练/回测)         │    │  │   (100万请求/月免费)        │
│  │                      │    │  │                            │
│  ├── Redis (25MB)       │    │  ├── AWS SNS                 │
│  │   (缓存/限流)         │    │  │   (100万推送/月免费)        │
│  │                      │    │  │                            │
│  └── Frontend           │    │  └── AWS Lambda              │
│      (React 静态站)      │    │      (100万次/月免费)         │
└─────────────────────────┘    └──────────────────────────────┘
```

> **成本：$0/月** — Render 免费 plan + AWS Free Tier + Upstash 免费层

### 已完成

- [x] 基础设施抽象层 `app/infra/`（broker / queue / notify / functions）
- [x] 每个接口 3 种后端实现（云服务 / Redis / In-Memory）
- [x] Settings 配置 + `.env.example` 文档

### 6. Redis 升级 — 从任务队列到全功能缓存层
- [x] 行情缓存：热门股票最新价格 Redis Hash（TTL 5s），减少 Polygon API 调用
- [x] Session 缓存：Paper Trading 持仓/订单状态存入 Redis
- [x] 分布式限流：`rate_limit.py` 改用 Redis 滑动窗口计数（替代内存计数）
- [x] Pub/Sub 价格推送：WebSocket 服务订阅 Redis channel，支持多实例广播
- [x] 模型 Artifact 缓存：预测时缓存已加载模型，避免重复从磁盘/S3 读取
- **部署：** Render 内置 Redis（25MB，免费）

### 7. SQS + SNS — 可靠消息队列 + 通知扇出
- [x] **SQS 替代 RQ：** 训练任务 → SQS 队列 → Worker 消费
  - 支持自动重试 + DLQ 死信队列（失败任务不丢失）
  - 可见性超时防止重复消费
- [x] **SNS 通知扇出：**
  - Alert topic：价格/指标/情绪预警 → 同时推送 Email + Discord + WebSocket
  - Training topic：训练完成 → 通知用户 + 触发模型评估
  - Signal topic：策略信号 → Paper Trading 引擎 + 日志记录
- **部署：** AWS Free Tier（SQS 100万/月 + SNS 100万/月）
- **本地开发：** LocalStack 模拟

### 8. Lambda — 无服务器事件处理
- [x] 新闻情绪分析：每条新闻触发 Lambda 调用 Claude Haiku 打分（天然并行）
- [x] Alert 触发器：价格/指标阈值突破 → Lambda 计算 → 发布到 SNS
- [x] 定时数据拉取：EventBridge 定时调度 → Lambda 拉取 Polygon 行情/新闻
- [x] SHAP 解释按需生成：API 请求 → Lambda 计算 SHAP 值 → 返回结果
- **部署：** AWS Free Tier（100万次/月 + 40万 GB-秒/月）
- **本地开发：** SAM CLI (`sam local invoke`)

### 9. Kafka — 实时数据管道
- [x] **行情流：** Polygon WebSocket → Kafka topic `market.prices` → 多消费者
- [x] **新闻流：** `news.raw` → 情绪分析消费者 → `news.scored`
- [x] **策略信号流：** 策略引擎订阅 prices → 产生信号写入 `signals.generated`
- [x] **回测回放：** Kafka 支持 offset 重置，可回放历史数据做回测验证
- [x] Schema Registry（Avro/Protobuf）保证消息格式一致性
- **部署：** Upstash Kafka Serverless（免费层 10,000 条/天）
- **本地开发：** Docker Compose 运行 Kafka（KRaft 模式，无 Zookeeper）

### 10. EKS / K8s（搁置，按需启用）
> 当前 Render 满足部署需求，EKS 作为未来扩展方案保留。
> 代码层已通过 Docker 容器化支持，随时可迁移到 K8s。
- [ ] Kubernetes manifests（Deployment / Service / ConfigMap / Secret）
- [ ] Helm Chart 一键部署
- [ ] HPA 自动扩缩
- [ ] Ingress Controller + TLS
- **备选方案：** Oracle Cloud OKE（永久免费 K8s）、GKE Autopilot
- **状态：** 🔒 搁置 — Render 够用时不启动，需要时再迁移

---

## 🤖 第三阶段：AI + 工程升级（Sub-projects）

### ✅ Sub 1 — Optuna 超参数优化（已完成 2026-04-10）
- [x] `Optuna` 多目标贝叶斯优化（NSGA-II），同时优化 Sharpe + MaxDD
- [x] 策略参数 TPE 搜索（均线周期、RSI 阈值、布林带 K）
- [x] 优化 Run 持久化：`optimization_runs` 表 + `/api/optimize/runs` 列表
- [x] 前端 Training 页 "Auto-Optimize" 按钮 + Strategy 页 "Optimize Params"

### ✅ Sub 2 — 多模型集成（已完成 2026-04-11）
- [x] `ModelFactory` 6 种模型：logistic / random forest / XGBoost / LightGBM / CatBoost / ensemble
- [x] Ensemble 支持 voting（软投票）+ stacking（KFold shuffle=False 保时序）
- [x] `/models/types` 暴露可用模型 + `available` 字段（可选依赖缺失时自动隐藏）
- [x] 置信度加权 + 模型分歧度作为风险信号

### ✅ Sub 3 — Distributed Systems（已完成 2026-04-16）
- [x] **Kubernetes manifests**（`k8s/` 目录 18 个 YAML）
  - API Deployment 2 副本 + HPA 2-5（CPU > 70% 自动扩容）
  - Consumer Deployment 独立 Pod（Kafka events → per-ticker stats）
  - Liveness / Readiness probes，失败自动重启
- [x] **Prometheus + Grafana** 观测栈
  - HTTP metrics 自动采集
  - 3 个自定义 ML metrics：`PREDICT_TOTAL` / `PREDICT_CONFIDENCE` / `MODEL_INFERENCE_SECONDS`
  - 预置 6 面板 Dashboard（RPS, p95, 预测数, 置信度热图, 推理时间, Pod 数）
- [x] **Kafka prediction event stream**
  - `/predict` 每次请求发布事件 → Consumer 聚合 `/stats/{ticker}`
  - `aiokafka` + in-memory fallback（pluggable via `BROKER_BACKEND`）
- [x] **Dockerfile.consumer** + `docker-compose.yml` 一键跑全栈（API + Consumer + Prometheus + Grafana）
- [x] **分布式文档**：`docs/architecture/distributed.md`（CAP 分析 + 生产扩容方案）
- [x] 274 unit + 45 contract tests 全绿，CI 无 skip
- **当前范围：** 本地 Minikube 演示（免费演练分布式）
- **后续：** Sub 5 — 云端托管 K8s（GKE Autopilot / EKS / OKE 免费层）

### ✅ Sub 4 — Frontend Redesign（已完成 2026-04-17）
> 从裸 inline styles 重构成企业级仪表盘 UI。

- [x] **新技术栈：**
  - UI kit：**Tremor** (图表/KPI) + **shadcn/ui** (built on Radix primitives)
  - 图表：**Lightweight Charts v4**（TradingView 同款，K 线 / 成交量）
  - 数据层：**TanStack Query v5**（cache / stale / auto-refetch）+ **Zustand**（WebSocket live store）
  - 表单：react-hook-form + zod 校验
  - 字体：Geist + Geist Mono（@font-face 加载 woff2）
  - 测试：Vitest + @testing-library/react（6 smoke tests）
- [x] **Dark theme 设计 token**（tailwind.config.js）— background/surface/accent/up/down/warn/info
- [x] **路由重构**：React Router v7 + AppShell + Sidebar（6 Lucide icons + Tooltip）
- [x] **6 个页面全部重写**（`src/pages/*Page.jsx`）：
  - Screener 排序表（点击跳 Dashboard）
  - Dashboard K 线 + Prediction + SHAP
  - Training 3-tab（Train / Runs / Models Promote）
  - Strategy（schema-driven params + backtest）
  - Trading（订单表 + 持仓卡 + 实时 WebSocket 价格）
  - Explain（SHAP top features + similar cases + graceful fallback）
- [x] **共享组件**：PageHeader / EmptyState / LoadingOverlay / ErrorBoundary / ConfirmDialog / TickerSearch
- [x] **CI frontend-test job**（npm ci → vitest --run → npm run build）
- [x] **legacy 清理**：删除 `Dashboard.jsx` / `Screener.jsx` 等 6 个旧页 + 3 个旧组件

### ✅ 数据层打通（已完成 2026-04-18）
- [x] **Supabase `prices` 表** — 程序化创建（SQLAlchemy `engine.begin()`），含 RLS（service_role 全权 + 公开读）
- [x] **yfinance 回填脚本**（`scripts/backfill_prices.py`）
  - 10 只股票（AAPL/MSFT/GOOGL/AMZN/NVDA/TSLA/META/JPM/V/WMT）× 2 年 = **5010 行**
  - 幂等 `ON CONFLICT DO NOTHING`，安全重跑
- [x] **前端数据 wiring 修 3 个 bug：**
  - `normalizeMarket` helper — 后端返回扁平数组，前端统一成 `{rows: [...]}`
  - `ScreenerPage` useMemo transform — 扁平行转成 per-ticker summary
  - `CandlestickChart` 切 hex — Lightweight Charts 拒绝 Tailwind v3 的空格分隔 `rgb(24 24 27)`

### ✅ 性能优化（已完成 2026-04-19）
- [x] **GitHub Actions keep-alive cron** — `.github/workflows/keepalive.yml`，每 10 分钟 ping `/health`，避免 Render 免费层 15 分钟冷启动
- [x] **Screener `lookback=5`** — 以前每 ticker 拉全量（2 年 × 10 只 ≈ 500KB），现在只拉最近 5 行 → **payload 减 98%**
- [x] **页面级代码拆分** — `React.lazy()` + `Suspense`，6 个页面各自 chunk → **首屏 JS 710KB → 335KB (-53%)**

### 🔜 Sub 5 — K8s 云端托管（Next sub-project）
> 把 `k8s/` 的本地 Minikube manifests 搬到托管 K8s，实现真正的弹性伸缩 + 生产可观测。
- [ ] 选平台：GKE Autopilot（推荐，$300 credit + per-pod billing）/ EKS（AWS 免费层 1 cluster）/ OKE（Oracle 永久免费 2 节点 ARM）
- [ ] Ingress + TLS（cert-manager + Let's Encrypt）
- [ ] Secret 管理（Sealed Secrets 或 External Secrets）
- [ ] 云端 Prometheus + Grafana（持久化 PVC）
- [ ] HPA 真实压测验证（k6 / locust）
- [ ] 迁移指南 `k8s/CLOUD_MIGRATION.md`

### ⏳ 待开发：ML/LLM 深化
- [ ] **强化学习 Agent**（`stable-baselines3`）— 状态空间：技术指标+情绪+持仓；奖励：Sharpe
- [ ] **LLM 策略生成** — Claude 读新闻 → 自然语言策略 → 自动转 Python 代码
- [ ] 策略 DSL（类 Pine Script 的简化语法）

---

## 🌐 第四阶段：平台化

### 15. 多市场支持
- [ ] 加密货币（Polygon Crypto API）
- [ ] ETF / 指数
- [ ] 美股期权（基础支持）

### 16. Alert 预警系统
- [ ] 价格突破提醒（通过 SNS 分发，第 2.5 阶段基础设施）
- [ ] 技术指标触发提醒（RSI 超买/超卖）
- [ ] 情绪异常预警（单日新闻情绪突变）
- [ ] 多渠道推送：Discord / Email / 前端通知

### 17. 策略社区（长期目标）
- [ ] 用户上传 Python 策略
- [ ] 公开回测结果排行榜
- [ ] 一键 fork 别人的策略
- [ ] 策略评分系统

### 18. 实盘对接（高级）
- [ ] 接 Alpaca API（美股，免佣金）
- [ ] 接 Interactive Brokers
- [ ] 风控模块（最大亏损止损、仓位限制）

### 19. 交互式图表工具（Chart Drawing Tools）

让用户像 TradingView 一样在 K 线图上手动画线分析，所有标注持久化存储。

基于 Lightweight Charts 已有基础，扩展交互绘图层：

**核心绘图工具：**
- [ ] 趋势线（Trendline）— 两点连线，自动延伸，吸附到 K 线 OHLC
- [ ] 水平线（Horizontal Line）— 标注关键价位，拖拽可调
- [ ] 支撑位 / 阻力位（Support / Resistance）— 水平区域高亮（半透明色带）
- [ ] 斐波那契回撤（Fibonacci Retracement）— 选起止点自动画 0% / 23.6% / 38.2% / 50% / 61.8% / 100% 水平线
- [ ] 垂直线 / 射线 / 平行通道（Vertical Line / Ray / Parallel Channel）
- [ ] 矩形区域标注（Rectangle）— 框选价格+时间区间，标注整理区 / 突破区

**图表交互体验：**
- [ ] 工具栏 UI：左侧浮动工具栏，图标切换绘图模式（类 TradingView 布局）
- [ ] 选中 / 编辑 / 删除已有标注（点击选中，拖拽调整，Delete 键删除）
- [ ] 标注颜色 / 线宽 / 样式自定义（实线 / 虚线 / 点线）
- [ ] 标注文字备注（可选：在线上附加文字标签，如 "突破位 $185"）
- [ ] Undo / Redo 支持（Ctrl+Z / Ctrl+Shift+Z）
- [ ] 十字准星 + 磁吸模式（绘图时自动吸附到最近 K 线的 OHLC 价格）

**数据持久化：**
- [ ] 前端 localStorage 缓存（即时保存，刷新不丢失）
- [ ] 后端 REST API 同步：`POST /api/drawings/{ticker}`, `GET /api/drawings/{ticker}`, `DELETE /api/drawings/{id}`
- [ ] 数据模型：`Drawing(id, ticker, type, points[], style, label, created_at, updated_at)`
- [ ] 切换股票时自动加载该 ticker 的所有标注

**与现有系统集成：**
- [ ] 策略信号叠加显示：买卖点标注（三角形箭头）+ 用户手绘线共存不冲突
- [ ] Screener 页跳转 Dashboard 时保留已有标注
- [ ] 回测结果可视化时，标注进出场点位 + 用户画线同屏
- [ ] 导出图表截图（含标注）功能（PNG / SVG）

**技术选型：**
- [ ] 基于 Lightweight Charts Drawing Primitives API
- [ ] 备选：自建 Canvas overlay 层（如 Drawing Primitives 不满足需求）
- [ ] React 组件封装：`<ChartWithDrawings ticker={ticker} />`

---

## 🔧 技术债务 & 工程优化

- [x] 完整单元测试（pytest，274 unit + 45 contract tests）
- [x] Frontend smoke tests（Vitest，6/6 pass）
- [x] CI/CD GitHub Actions（push 自动测试 + lint，无 skip，无 continue-on-error）
- [x] **监控 & 可观测性**（Prometheus `/metrics` + Grafana 6 面板 Dashboard）✅ Sub 3
- [x] **前端性能优化**（keep-alive cron + 代码拆分 + Screener lookback 收缩）✅ 2026-04-19
- [x] Docker Compose 一键启动（API + Consumer + Prometheus + Grafana + DB）
- [ ] API 文档完善（OpenAPI examples）
- [ ] 前端 E2E 测试（Playwright）
- [ ] Terraform / CDK 基础设施即代码（GKE/EKS 资源管理）
- [ ] 日志聚合（Loki / CloudWatch Logs）
- [ ] 定时数据拉取（GitHub Actions cron → `backfill_prices.py` daily after US close）

---

## 🏗️ 架构演进

```
Phase 3 Sub 3/4 (当前):

  Vercel CDN (React 19 + Tremor + shadcn + Lightweight Charts)
       │
       │ HTTPS + WebSocket
       ▼
  Render Backend (FastAPI + /metrics)
       │
       ├─────────────┬──────────────┬───────────────┐
       ▼             ▼              ▼               ▼
  Supabase PG   Render Redis   Kafka Events   AWS Lambda
  (prices=5010)  (cache/WS)    (aiokafka)     (情感分析)
       │             │              │
       │             │              ▼
       │             │         Consumer Pod
       │             │         (/stats/{ticker})
       │             │
       ▼             ▼
  RLS 公开读    分布式限流

本地 Minikube (k8s/*.yaml, Sub 3):
  ┌───────────────┬───────────────┬──────────┬──────────┐
  │ api pod ×2    │ consumer pod  │Prometheus│ Grafana  │
  │ HPA 2-5       │ stats aggreg. │ scrape   │ 6 panels │
  └───────────────┴───────────────┴──────────┴──────────┘

GitHub Actions:
  · CI (test + lint + build + docker)
  · keep-alive cron */10min (ping /health)
```

---

## 💰 月度成本

| 服务 | 提供商 | 免费额度 | 预估费用 |
|------|--------|---------|---------|
| API + Worker + Frontend | Render | Free plan | $0 |
| Redis (25MB) | Render | 内置免费 | $0 |
| Kafka | Upstash | 10,000 msg/天 | $0 |
| SQS | AWS | 100万请求/月 | $0 |
| SNS | AWS | 100万推送/月 | $0 |
| Lambda | AWS | 100万次/月 | $0 |
| **总计** | | | **$0/月** |

---

## 项目信息

**项目地址：** https://github.com/jigangz/quant-ai
**主分支：** main
**Live 部署：**
- Frontend: https://quant-ai-ui.vercel.app (Vercel)
- Backend API: https://quant-ai-qzrg.onrender.com (Render)
- API Docs: https://quant-ai-qzrg.onrender.com/docs
- Database: Supabase `ppxkpookjsbqfxsjxfck`（`prices` + `news` 表 + RLS）

**技术栈：**
- **Backend:** Python 3.11 + FastAPI + SQLAlchemy + Pydantic + Prometheus client
- **Frontend:** React 19 + Vite + Tailwind v3 + Tremor + shadcn/ui（Radix） + Lightweight Charts v4 + TanStack Query v5 + Zustand + react-hook-form + zod + Geist fonts + Vitest
- **Data:** PostgreSQL (Supabase) / Redis / Kafka (aiokafka) / SQS / S3
- **ML:** scikit-learn / XGBoost / LightGBM / CatBoost / Optuna / SHAP
- **Infra:** Docker multi-stage + Kubernetes (k8s/) + Prometheus + Grafana

**已知技术决策：**
- Python 策略类替代 Pine Script
- Lightweight Charts 替代 D3（Phase 1 完成）
- Optuna 做参数优化（Phase 3 Sub 1 完成）
- 前端彻底重构成 Tremor + shadcn（Phase 3 Sub 4 完成 2026-04-17）
- 本地 Minikube 完整跑 K8s（Phase 3 Sub 3 完成 2026-04-16），云端托管留给 Sub 5
- yfinance 替代 Polygon（开发期免费），生产期如需实时切 Polygon

---

*最后更新：2026-04-19*
*下次继续：前端逐页功能检查 → Sub 5 K8s 云端托管*
