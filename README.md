# Quant AI Backend

A data-driven quantitative research and prediction platform.

**面试官 2 分钟看懂这个项目：**
1. 🎯 **做什么**: 用机器学习预测股票涨跌方向
2. 🔧 **怎么做**: FastAPI + 可插拔 ML 模型 + 回测引擎
3. ⚠️ **核心难点**: 防止数据泄漏 + 模型版本化 + 策略评估

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ /health  │ │  /train  │ │ /predict │ │/backtest │ │ /explain │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────┘
        │            │            │            │            │
┌───────┼────────────┼────────────┼────────────┼────────────┼─────────┐
│       │     Service Layer       │            │            │         │
│       │    ┌────────────────────┴───┐  ┌─────┴─────┐  ┌───┴────┐   │
│       │    │   TrainingService      │  │ Backtest  │  │  SHAP  │   │
│       │    │ - DatasetBuilder       │  │  Engine   │  │Explainer│  │
│       │    │ - ModelFactory         │  └───────────┘  └────────┘   │
│       │    │ - Train + Evaluate     │                              │
│       │    └────────────────────────┘                              │
└───────┼─────────────┬───────────────────────────────────────────────┘
        │             │
┌───────┼─────────────┼───────────────────────────────────────────────┐
│       │    ML Layer │                                               │
│  ┌────┴────┐  ┌─────┴─────┐  ┌────────────┐  ┌─────────────────┐   │
│  │ Feature │  │  Model    │  │   Label    │  │   Time-Series   │   │
│  │Registry │  │  Factory  │  │  Generator │  │     Splitter    │   │
│  │(Groups) │  │(Logistic/ │  │(Direction/ │  │  (No Leakage!)  │   │
│  │         │  │ XGBoost)  │  │  Returns)  │  │                 │   │
│  └─────────┘  └───────────┘  └────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│       │         Data Layer                                          │
│  ┌────┴────────┐  ┌────────────────┐  ┌─────────────────────────┐  │
│  │   Market    │  │     Model      │  │      Artifacts          │  │
│  │  Provider   │  │   Registry     │  │    (local/S3)           │  │
│  │  (Yahoo)    │  │  (Supabase)    │  │                         │  │
│  └─────────────┘  └────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Data Leak Prevention

**问题**: 时序数据如果随机 shuffle 切分，会把"未来"数据混进训练集 → 模型在生产环境失效。

**解决方案**: `DatasetBuilder._time_series_split()` 按日期切分，不 shuffle：

```python
# ❌ 错误: 随机切分
X_train, X_test = train_test_split(X, shuffle=True)  # 未来数据泄漏!

# ✅ 正确: 按时间顺序切分
# 2020-01-01 ~ 2023-06-30 → 训练集
# 2023-07-01 ~ 2023-09-30 → 验证集  
# 2023-10-01 ~ 2024-01-01 → 测试集
```

**关键代码** (`app/ml/dataset/builder.py`):
```python
def _time_series_split(self, df):
    # Split by DATE, not by row index
    unique_dates = df["date"].unique()
    train_end_date = unique_dates[int(len(unique_dates) * 0.7)]
    
    train_df = df[df["date"] <= train_end_date]
    # ... val and test follow sequentially
```

---

## 📦 Model Versioning

**目标**: 追踪每个模型的来源、参数、性能，支持回滚。

**存储结构**:
```
artifacts/
├── logistic_AAPL_20240131_143022/
│   ├── model.joblib        # 模型权重
│   ├── metadata.json       # 训练参数、特征列表
│   └── metrics.json        # 评估指标
└── xgboost_AAPL_MSFT_20240201_091500/
    └── ...
```

**Model Registry** (`app/db/model_registry.py`):
```python
class ModelRecord(BaseModel):
    id: str           # UUID
    name: str         # "logistic_AAPL_20240131"
    version: int      # 自增版本号
    model_type: str   # "logistic" | "xgboost"
    tickers: list     # ["AAPL", "MSFT"]
    feature_groups: list  # ["ta_basic", "momentum"]
    metrics: dict     # {"accuracy": 0.56, "auc": 0.62}
    artifact_path: str    # 本地或 S3 路径
    created_at: datetime
```

**支持的存储后端**:
- `local`: 本地文件系统 (开发)
- `supabase`: Supabase Storage (生产)

---

## 📊 Backtest Evaluation

**回测流程**:
```
1. 加载模型 → 2. 生成预测 → 3. 模拟交易 → 4. 计算指标 → 5. 对比 Buy & Hold
```

**分类指标** (Classification Metrics):
| 指标 | 含义 | 目标 |
|------|------|------|
| AUC | 模型区分能力 | > 0.55 |
| F1 | 精确率和召回率的调和平均 | > 0.50 |
| Precision | 预测涨时真正涨的比例 | 高 |
| Recall | 真正涨时被预测到的比例 | 高 |

**策略指标** (Strategy Metrics):
| 指标 | 含义 | 目标 |
|------|------|------|
| CAGR | 年化复合收益率 | > Buy & Hold |
| Sharpe Ratio | 风险调整后收益 | > 1.0 |
| Max Drawdown | 最大回撤 | < 20% |
| Win Rate | 盈利交易占比 | > 50% |
| Profit Factor | 总盈利/总亏损 | > 1.5 |

**API 示例**:
```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"model_id": "abc123", "signal_threshold": 0.55}'
```

---

## Version Overview

| Version | Focus | Status |
|---------|-------|--------|
| **V1** | Data collection, feature engineering, baseline model, SHAP explainability | ✅ Complete |
| **V2** | Multi-ticker, multi-model, training API, model registry, backtesting | ✅ Complete (10 batches) |
| **V3** | Async training, experiment tracking, UI training panel, RAG | 📋 Planned |

---

## Quick Start

### 🐳 Docker (Recommended)

```bash
git clone https://github.com/jigangz/quant-ai.git
cd quant-ai
cp .env.example .env
docker-compose up
```

### 🐍 Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

### ✅ Verify

```bash
curl http://localhost:8000/health
```

---

## 🎬 Demo Scripts

### 30 秒快速演示

```bash
python scripts/demo_30s.py
```

展示: Health check → 列出已有模型

### 2 分钟完整演示

```bash
python scripts/demo_2min.py
```

展示: 训练 → 模型注册 → 回测 → 结果分析

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | 服务状态 + 配置信息 |
| `/data/market` | GET | 获取市场数据 |
| `/train` | POST | 训练模型 |
| `/models` | GET | 列出所有模型 |
| `/models/{id}` | GET | 获取模型详情 |
| `/predict` | POST | 模型预测 |
| `/backtest` | POST | 运行回测 |
| `/explain` | GET | SHAP 解释 |

---

## Project Structure

```
quant-ai/
├── app/
│   ├── api/              # FastAPI routes
│   │   ├── train.py      # POST /train
│   │   ├── backtest.py   # POST /backtest
│   │   └── ...
│   ├── core/
│   │   └── settings.py   # Pydantic Settings
│   ├── ml/
│   │   ├── dataset/      # DatasetBuilder + schemas
│   │   ├── features/     # FeatureRegistry + groups
│   │   └── models/       # ModelFactory + implementations
│   ├── backtest/
│   │   ├── engine.py     # BacktestEngine
│   │   └── metrics.py    # CAGR, Sharpe, etc.
│   ├── db/
│   │   └── model_registry.py  # Model versioning
│   ├── providers/        # Data providers
│   └── services/         # Business logic
├── scripts/
│   ├── demo_30s.py       # Quick demo
│   ├── demo_2min.py      # Full demo
│   └── train.py          # Legacy training script
├── artifacts/            # Model storage
├── docker-compose.yml
└── README.md
```

---

## Configuration

See `.env.example` for all options.

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment | `dev` |
| `DEFAULT_MODEL_TYPE` | Model type | `logistic` |
| `DEFAULT_FEATURE_GROUPS` | Feature groups | `ta_basic,momentum` |
| `STORAGE_BACKEND` | Artifact storage | `local` |
| `SUPABASE_URL` | Supabase URL (optional) | - |

---

## License

MIT
