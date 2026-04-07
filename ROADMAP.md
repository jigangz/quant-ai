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

## 🤖 第三阶段：AI 优化量化策略

### 11. ML 优化策略参数
- [ ] **贝叶斯优化**（Optuna）自动搜索最优参数
  - 例：自动找最优均线周期、RSI 阈值
- [ ] **遗传算法**优化策略组合
- [ ] 参数优化结果可视化（热力图）

### 12. 强化学习交易智能体（RL Agent）
- [ ] 接入 `stable-baselines3`
- [ ] 状态空间：技术指标 + 新闻情绪 + 持仓状态
- [ ] 动作空间：买入 / 卖出 / 持有
- [ ] 奖励函数：风险调整后收益（Sharpe 导向）
- [ ] 对比 XGBoost vs RL 策略回测表现

### 13. 多模型集成预测
- [ ] Ensemble：XGBoost + LSTM + Transformer 投票
- [ ] 置信度加权，高置信度才下单
- [ ] 模型分歧度作为风险信号（分歧大 = 不确定性高）

### 14. LLM 策略生成（实验性）
- [ ] 用 Claude 分析新闻，生成自然语言策略描述
- [ ] 自动转换为 Python 策略代码
- [ ] 类 Pine Script 的简化 DSL（quant-ai 自己的策略语言）

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

- [ ] 完整单元测试（pytest，覆盖率 > 80%）
- [ ] Docker Compose 一键启动文档（含 Kafka + LocalStack）
- [ ] CI/CD GitHub Actions（push 自动测试 + lint）
- [ ] API 文档完善（OpenAPI examples）
- [ ] 前端 E2E 测试（Playwright）
- [ ] 性能优化：大量历史数据的查询缓存
- [ ] Terraform / CDK 基础设施即代码（AWS 资源管理）
- [ ] 监控 & 可观测性（Prometheus + Grafana / CloudWatch）
- [ ] 日志聚合（ELK / CloudWatch Logs）

---

## 🏗️ 架构演进

```
Phase 1-2 (当前):
  Client → Render (FastAPI) → SQLite/Supabase
                             → Redis (RQ 任务队列)
                             → WebSocket (mock 价格)

Phase 2.5 (目标):
  Client → Render
              │
         ┌────┴────┐
         │ API Pod │────→ Render Redis (缓存/限流/Pub-Sub)
         └────┬────┘
              │
         ┌────┴────────┐
         │ Upstash     │ (实时行情/新闻/信号流)
         │ Kafka       │
         └────┬────────┘
              │
    ┌─────────┼─────────┐
    ↓         ↓         ↓
┌───────┐ ┌───────┐ ┌────────┐
│Render │ │AWS    │ │Render  │
│Worker │ │Lambda │ │Stream  │
│(SQS)  │ │(事件) │ │Service │
└───────┘ └───┬───┘ └────────┘
              ↓
         ┌────────┐
         │AWS SNS │ → Email / Discord / WebSocket
         └────────┘
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
**部署：** Render (API/Worker/Frontend/Redis) + Upstash Kafka + AWS Free Tier (SQS/SNS/Lambda)
**技术栈：** Python (FastAPI) + React + PostgreSQL/SQLite + Redis + Kafka + AWS

**已知技术决策：**
- Python 策略类替代 Pine Script
- Lightweight Charts 替代 D3（已完成）
- Optuna 做参数优化
- stable-baselines3 做 RL 智能体
- Render 做主要部署平台（免费）
- Upstash Kafka 替代 AWS MSK（免费 Serverless）
- LocalStack 模拟 AWS 服务（本地开发）
- EKS 搁置，Render 够用时不迁移

---

*最后更新：2026-04-02*
*下次继续：第三阶段 — ML 优化策略参数 + RL 交易智能体*
