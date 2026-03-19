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

## 🤖 第三阶段：AI 优化量化策略（重点方向）

### 6. ML 优化策略参数
- [ ] **贝叶斯优化**（Optuna）自动搜索最优参数
  - 例：自动找最优均线周期、RSI 阈值
- [ ] **遗传算法**优化策略组合
- [ ] 参数优化结果可视化（热力图）

### 7. 强化学习交易智能体（RL Agent）
- [ ] 接入 `stable-baselines3`
- [ ] 状态空间：技术指标 + 新闻情绪 + 持仓状态
- [ ] 动作空间：买入 / 卖出 / 持有
- [ ] 奖励函数：风险调整后收益（Sharpe 导向）
- [ ] 对比 XGBoost vs RL 策略回测表现

### 8. 多模型集成预测
- [ ] Ensemble：XGBoost + LSTM + Transformer 投票
- [ ] 置信度加权，高置信度才下单
- [ ] 模型分歧度作为风险信号（分歧大 = 不确定性高）

### 9. LLM 策略生成（实验性）
- [ ] 用 Claude 分析新闻，生成自然语言策略描述
- [ ] 自动转换为 Python 策略代码
- [ ] 类 Pine Script 的简化 DSL（quant-ai 自己的策略语言）

---

## 🌐 第四阶段：平台化

### 10. 多市场支持
- [ ] 加密货币（Polygon Crypto API）
- [ ] ETF / 指数
- [ ] 美股期权（基础支持）

### 11. Alert 预警系统
- [ ] 价格突破提醒
- [ ] 技术指标触发提醒（RSI 超买/超卖）
- [ ] 情绪异常预警（单日新闻情绪突变）
- [ ] 推送到 Discord / 邮件

### 12. 策略社区（长期目标）
- [ ] 用户上传 Python 策略
- [ ] 公开回测结果排行榜
- [ ] 一键 fork 别人的策略
- [ ] 策略评分系统

### 13. 实盘对接（高级）
- [ ] 接 Alpaca API（美股，免佣金）
- [ ] 接 Interactive Brokers
- [ ] 风控模块（最大亏损止损、仓位限制）

---

## 技术债务 & 工程优化

- [ ] 完整单元测试（pytest，覆盖率 > 80%）
- [ ] Docker Compose 一键启动文档
- [ ] CI/CD GitHub Actions（push 自动测试）
- [ ] API 文档完善（OpenAPI examples）
- [ ] 前端 E2E 测试（Playwright）
- [ ] 性能优化：大量历史数据的查询缓存

---

## 当前 Session 上下文

**项目地址：** https://github.com/jigangz/quant-ai
**主分支：** main
**本地路径（开发用）：** /tmp/quant-ai
**前端路径：** /tmp/quant-ai/frontend
**参考项目：** /tmp/PokieTicker（新闻可视化灵感来源）

**环境变量：**
- `POLYGON_API_KEY` — 已配置
- `ANTHROPIC_API_KEY` — 暂时跳过（情绪分析可选）
- `DATABASE_URL` — PostgreSQL（本地开发可用 SQLite）

**已知技术决策：**
- Python 策略类替代 Pine Script
- Lightweight Charts 替代 D3（已完成）
- Optuna 做参数优化
- stable-baselines3 做 RL 智能体

---

*最后更新：2026-03-19*
*下次继续：第三阶段 — ML 优化策略参数 + RL 交易智能体*
