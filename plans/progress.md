# Progress Log

## 2026-04-13

### Session Start
- Brainstorming complete, design spec approved
- Implementation plan written (17 tasks, 3 phase gates)
- Ralph infra set up, prd.json created
- Starting automated execution

### Batch 1 (P1-1, P1-2, P1-3) — completed 2026-04-14

**P1-1: explain_service.py lazy load**
- Changed from eager `_explainer = ShapExplainer(model_path=...)` to lazy `_get_explainer()` with global None
- Tests use `patch.object(explain_mod, "ShapExplainer", ...)` + `setup_method` to reset global (plan code used `reload` which breaks patches)
- sklearn and other deps needed `pip install -r requirements.txt` first

**P1-2: CI test infrastructure**
- Changed `asyncio_mode = "strict"` → `"auto"` in pyproject.toml
- Added `_ensure_tables` autouse session fixture in both `tests/conftest.py` and `tests/contract/conftest.py`
- Changed `DATABASE_URL` from `sqlite:///:memory:` to `sqlite:///./test_quant.db` (in-memory creates new DB per connection)
- Pre-import critical modules in conftest to prevent `test_functions.py` from mocking them via `sys.modules.setdefault`
- Fixed `app/api/train.py`: replaced `asyncio.get_event_loop()` (deprecated in 3.10+) with `asyncio.get_running_loop()` + try/except
- Fixed `app/backtest/engine.py`: changed `position_size` field `le=2.0` → `le=1.0` (contract test expected 422 for 2.0)
- Fixed `app/api/predict.py`: removed stray `lookback=500` kwarg that `PredictionService.predict()` doesn't accept
- Fixed `app/api/train.py`: added model_type validation before queuing (returns 400 for invalid types)
- Fixed `tests/contract/conftest.py`: added missing `sample_predict_request` and `sample_backtest_request` fixtures
- Updated contract tests to match actual async API response format

**P1-3: API tests**
- Created `tests/test_api_market.py` (3 tests)
- Created `tests/test_api_features.py` (4 tests)
- Created `tests/test_api_models.py` (3 tests)
- Fixed `app/api/models.py` route ordering: moved `/models/cache` and `/models/promoted` routes BEFORE `/{model_id}` to prevent parameterized route shadowing
- All 203 unit tests + 39 contract tests pass

### Batch 2 (P1-4) — completed 2026-04-13

**P1-4: API tests for explain, search, agents, news**
- Created `tests/test_api_explain.py` (2 tests)
- Created `tests/test_api_search.py` (2 tests)
- Created `tests/test_api_agents.py` (3 tests)
- Created `tests/test_api_news.py` (3 tests)
- Fixed `app/api/agents.py`: `PortfolioSummaryResponse.overall_signal` was required with no default — made optional (None) so failure path works. `bullish_count`/`bearish_count` given default 0.
- Plan code used `patch("app.api.agents.get_model_cache")` but get_model_cache is a local import inside the function, so correct target is `app.services.model_cache.get_model_cache`
- 213 unit tests pass

### Batch 3 (P1-GATE) — completed 2026-04-13

**P1-GATE: Phase 1 gate verification**
- Unit tests: 213 passed, 0 failures
- Contract tests: 39 passed, 0 failures
- ruff check app/ scripts/ --ignore F401,F841,E501,F541,E402: all checks passed
- CI yml clean: no continue-on-error on test steps, no --ignore workarounds for test files, contract tests run as separate step
- Dockerfile valid with `--target production` stage, Docker Desktop not available locally (will verify in CI)
- Gate passed, Phase 2 (frontend) can proceed

### Batch 4 (P2-1, P2-2, P2-3) — completed 2026-04-14

**P2-1: Install Tailwind CSS + React Router**
- Installed react-router-dom@7.14.1, tailwindcss@3.4.19 (v3, not v4), postcss, autoprefixer
- Plan listed `@tailwindcss/vite` but used standard v3 postcss approach (config format in plan is v3-style)
- Created tailwind.config.js with dark theme + custom colors (surface, accent, up, down)
- Created postcss.config.js with tailwindcss + autoprefixer plugins
- Replaced index.css with @tailwind directives + bg-surface body style
- Wrapped App with BrowserRouter in main.jsx
- npm run build: 0 errors

**P2-2: Rewrite App.jsx with React Router**
- Replaced page-state pattern with React Router Routes/Route/NavLink
- 6-page nav: Screener, Dashboard, Training, Strategy, Trading, Explain
- Active nav link uses bg-accent styling, inactive uses text-gray-400
- Deleted empty stubs: PriceChart.jsx, PredictionCard.jsx, ShapList.jsx
- Created placeholder pages for Screener/Strategy/Trading
- npm run build: 0 errors

**P2-3: Add API client functions**
- Appended to client.js: listStrategies, getStrategy, generateSignals, runStrategyBacktest
- Appended: placeOrder, listOrders, cancelOrder, getPortfolio, getPortfolioHistory, resetPortfolio, getTrades
- Appended: getMarketMulti (uses Promise.all over getMarket)
- npm run build: 0 errors
