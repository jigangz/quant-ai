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

## ☁️ 第 2.5 阶段：云原生基础设施

> 将单体架构升级为 AWS 云原生架构，引入消息队列、事件驱动和容器编排。
> 开发阶段全部本地 Docker 运行，零云端费用。

### 6. Redis 升级 — 从任务队列到全功能缓存层
- [ ] 行情缓存：热门股票最新价格 Redis Hash（TTL 5s），减少 Polygon API 调用
- [ ] Session 缓存：Paper Trading 持仓/订单状态存入 Redis
- [ ] 分布式限流：`rate_limit.py` 改用 Redis 滑动窗口计数（替代内存计数）
- [ ] Pub/Sub 价格推送：WebSocket 服务订阅 Redis channel，支持多实例广播
- [ ] 模型 Artifact 缓存：预测时缓存已加载模型，避免重复从磁盘/S3 读取

### 7. SQS + SNS — 可靠消息队列 + 通知扇出
- [ ] **SQS 替代 RQ：** 训练任务 → SQS 队列 → Worker 消费
  - 支持自动重试 + DLQ 死信队列（失败任务不丢失）
  - 可见性超时防止重复消费
- [ ] **SNS 通知扇出：**
  - Alert topic：价格/指标/情绪预警 → 同时推送 Email + Discord + WebSocket
  - Training topic：训练完成 → 通知用户 + 触发模型评估
  - Signal topic：策略信号 → Paper Trading 引擎 + 日志记录
- [ ] **本地开发：** 使用 LocalStack 或 ElasticMQ 模拟 SQS/SNS

### 8. Lambda — 无服务器事件处理
- [ ] 新闻情绪分析：每条新闻触发 Lambda 调用 Claude Haiku 打分（天然并行）
- [ ] Alert 触发器：价格/指标阈值突破 → Lambda 计算 → 发布到 SNS
- [ ] 定时数据拉取：EventBridge 定时调度 → Lambda 拉取 Polygon 行情/新闻
- [ ] SHAP 解释按需生成：API 请求 → Lambda 计算 SHAP 值 → 返回结果
- [ ] **本地开发：** SAM CLI (`sam local invoke`) 本地测试

### 9. Kafka — 实时数据管道
- [ ] **行情流：** Polygon WebSocket → Kafka topic `market.prices` → 多消费者
- [ ] **新闻流：** `news.raw` → 情绪分析消费者 → `news.scored`
- [ ] **策略信号流：** 策略引擎订阅 prices → 产生信号写入 `signals.generated`
- [ ] **回测回放：** Kafka 支持 offset 重置，可回放历史数据做回测验证
- [ ] Schema Registry（Avro/Protobuf）保证消息格式一致性
- [ ] **本地开发：** Docker Compose 运行 Kafka + Zookeeper（或 KRaft 模式）

### 10. EKS — 容器编排 + 微服务部署
- [ ] **服务拆分：**
  - `api-service` — FastAPI 主 API（HPA 自动扩缩）
  - `worker-service` — 训练/回测 Worker（CPU 密集，独立扩缩）
  - `streaming-service` — Kafka 消费者 + WebSocket 推送
  - `frontend` — React 静态资源（Nginx）
- [ ] Kubernetes manifests（Deployment / Service / ConfigMap / Secret）
- [ ] Helm Chart 一键部署整个技术栈
- [ ] HPA 自动扩缩：根据 CPU/内存/队列深度弹性伸缩 Worker
- [ ] Ingress Controller（ALB Ingress）统一入口 + TLS
- [ ] **本地开发：** minikube 或 kind 运行本地 K8s 集群

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
- [ ] 价格突破提醒（通过 SNS 分发）
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

---

## 🔧 技术债务 & 工程优化

- [ ] 完整单元测试（pytest，覆盖率 > 80%）
- [ ] Docker Compose 一键启动文档（含全套基础设施）
- [ ] CI/CD GitHub Actions（push 自动测试 + lint）
- [ ] API 文档完善（OpenAPI examples）
- [ ] 前端 E2E 测试（Playwright）
- [ ] 性能优化：大量历史数据的查询缓存
- [ ] Terraform / CDK 基础设施即代码（IaC）
- [ ] 监控 & 可观测性（Prometheus + Grafana / CloudWatch）
- [ ] 日志聚合（ELK / CloudWatch Logs）

---

## 🏗️ 架构演进

```
当前 (v2):
  Client → FastAPI → PostgreSQL
                   → Redis (RQ)
                   → WebSocket (mock prices)

目标 (v2.5+):
  Client → ALB/Ingress
              ↓
         ┌─────────┐     ┌─────────┐
         │ API Pod │────→│  Redis  │ (缓存/限流/Pub-Sub)
         └────┬────┘     └─────────┘
              │
         ┌────┴────┐
         │  Kafka  │ (实时行情/新闻/信号流)
         └────┬────┘
              │
    ┌─────────┼─────────┐
    ↓         ↓         ↓
┌───────┐ ┌───────┐ ┌────────┐
│Worker │ │Lambda │ │Stream  │
│(SQS)  │ │(事件) │ │Service │
└───────┘ └───┬───┘ └────────┘
              ↓
         ┌────────┐
         │  SNS   │ → Email / Discord / WebSocket
         └────────┘
```

---

## 💰 开发成本

| 服务 | 本地开发 | AWS 免费层 | 超出后 |
|------|---------|-----------|--------|
| Redis | Docker ✅ 免费 | — | ElastiCache ~$13/月 |
| SQS | LocalStack ✅ 免费 | 100万请求/月 | 几乎用不完 |
| SNS | LocalStack ✅ 免费 | 100万推送/月 | 几乎用不完 |
| Lambda | SAM CLI ✅ 免费 | 100万次/月 | 个人项目够用 |
| Kafka | Docker ✅ 免费 | — | MSK ~$150/月 |
| EKS | minikube ✅ 免费 | — | 控制面 $73/月 + EC2 |

> **策略：本地 Docker 全套开发，Portfolio 展示架构设计和代码实现。生产部署方案作为文档存在即可。**

---

## 项目信息

**项目地址：** https://github.com/jigangz/quant-ai
**主分支：** main
**技术栈：** Python (FastAPI) + React + PostgreSQL + Redis + Kafka + AWS (Lambda/SQS/SNS/EKS)

**已知技术决策：**
- Python 策略类替代 Pine Script
- Lightweight Charts 替代 D3（已完成）
- Optuna 做参数优化
- stable-baselines3 做 RL 智能体
- LocalStack 模拟 AWS 服务（本地开发）
- minikube 模拟 EKS（本地开发）

---

*最后更新：2026-04-01*
*下次继续：第 2.5 阶段 — Redis 升级 → SQS+SNS → Lambda → Kafka → EKS*
