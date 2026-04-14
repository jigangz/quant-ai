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
