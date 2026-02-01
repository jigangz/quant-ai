#!/usr/bin/env python3
"""
V3 Full Demo - Showcases all major features.

Steps:
1. Train multiple models (Logistic, XGBoost, LightGBM)
2. Compare backtest results
3. Promote best model to production
4. Predict with production model
5. Technical analysis with SHAP
6. RAG Q&A

Usage:
    python scripts/demo_v3.py
    python scripts/demo_v3.py --base-url http://your-server:8000
    python scripts/demo_v3.py --quick  # use existing models
"""

import argparse
import json
import sys
import time
from typing import Any

import requests

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def print_header(msg: str):
    print(f"\n{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{CYAN}{'═' * 70}{RESET}\n")


def print_step(n: int, msg: str):
    print(f"\n{MAGENTA}▶ Step {n}: {msg}{RESET}")
    print(f"{DIM}{'─' * 50}{RESET}")


def print_ok(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")


def print_warn(msg: str):
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def print_error(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON snippet."""
    formatted = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    lines = formatted.split("\n")
    for line in lines[:15]:
        print(f"  {DIM}{line}{RESET}")
    if len(lines) > 15:
        print(f"  {DIM}... ({len(lines) - 15} more lines){RESET}")


def api_call(base_url: str, method: str, path: str, data: dict = None, timeout: int = 120) -> dict:
    """Make API call and return response."""
    url = f"{base_url}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        elif method == "POST":
            resp = requests.post(url, json=data, timeout=timeout)
        elif method == "DELETE":
            resp = requests.delete(url, timeout=timeout)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        try:
            return {"error": resp.json()}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def demo(base_url: str, quick: bool = False):
    """Run the full V3 demo."""
    
    print_header("🚀 Quant AI V3 - Interview Demo")
    print(f"  Base URL: {base_url}")
    print(f"  Quick mode: {quick}")
    
    start_time = time.time()
    trained_models = []
    backtest_results = []
    
    # =========================================
    # Step 1: Health Check
    # =========================================
    print_step(1, "Health Check")
    
    result = api_call(base_url, "GET", "/health")
    if "error" in result:
        print_error(f"Cannot connect: {result['error']}")
        print("  Make sure the server is running: docker-compose up")
        return False
    
    print_ok(f"Status: {result.get('status', 'ok')}")
    settings = result.get("settings", {})
    print(f"  • Environment: {settings.get('env', 'N/A')}")
    print(f"  • Default model: {settings.get('default_model_type', 'N/A')}")
    
    # =========================================
    # Step 2: Train Multiple Models
    # =========================================
    print_step(2, "Train Multiple Models (Logistic, XGBoost, LightGBM)")
    
    if quick:
        print_warn("Quick mode - checking existing models...")
        models_result = api_call(base_url, "GET", "/models?limit=10")
        if models_result.get("models"):
            for m in models_result["models"][:3]:
                trained_models.append({
                    "model_id": m["id"],
                    "model_type": m["model_type"],
                    "metrics": m.get("metrics", {}),
                })
            print_ok(f"Found {len(trained_models)} existing models")
        else:
            print_warn("No models found, will train new ones")
            quick = False
    
    if not quick:
        model_configs = [
            {
                "name": "Logistic Regression",
                "type": "logistic",
                "params": {},
            },
            {
                "name": "XGBoost",
                "type": "xgboost",
                "params": {"n_estimators": 50, "max_depth": 4},
            },
            {
                "name": "LightGBM",
                "type": "lightgbm",
                "params": {"n_estimators": 50, "num_leaves": 15},
            },
        ]
        
        for config in model_configs:
            print(f"\n  Training {config['name']}...")
            
            train_request = {
                "tickers": ["AAPL", "MSFT"],
                "feature_groups": ["ta_basic", "momentum"],
                "model_type": config["type"],
                "model_params": config["params"],
                "horizon_days": 5,
                "search_mode": "none",
            }
            
            result = api_call(base_url, "POST", "/train?async=false", train_request)
            
            if "error" not in result and result.get("id"):
                model_id = result["id"]
                metrics = result.get("metrics", {})
                trained_models.append({
                    "model_id": model_id,
                    "model_type": config["type"],
                    "name": config["name"],
                    "metrics": metrics,
                })
                val_auc = metrics.get("val_auc", "N/A")
                print_ok(f"{config['name']}: val_auc={val_auc}")
            else:
                print_warn(f"{config['name']} failed: {result.get('error', 'Unknown')}")
    
    if not trained_models:
        print_error("No models trained. Cannot continue demo.")
        return False
    
    # =========================================
    # Step 3: Run Backtests & Compare
    # =========================================
    print_step(3, "Run Backtests & Compare Performance")
    
    for model in trained_models:
        print(f"\n  Backtesting {model.get('name', model['model_type'])}...")
        
        backtest_request = {
            "model_id": model["model_id"],
            "tickers": ["AAPL", "MSFT"],
            "position_sizing": "volatility_scaled",
            "enable_costs": True,
            "enable_slippage": True,
            "portfolio_weighting": "equal",
        }
        
        result = api_call(base_url, "POST", "/backtest", backtest_request)
        
        if result.get("success"):
            metrics = result.get("strategy_metrics", {})
            model["backtest"] = {
                "sharpe": metrics.get("sharpe", 0),
                "total_return": metrics.get("total_return", 0),
                "max_drawdown": result.get("max_drawdown", 0),
                "n_trades": metrics.get("n_trades", 0),
            }
            backtest_results.append({
                "model": model,
                "equity_curve": result.get("equity_curve", []),
            })
            print_ok(f"Sharpe: {metrics.get('sharpe', 0):.2f}, Return: {metrics.get('total_return', 0):.1f}%")
        else:
            print_warn(f"Backtest failed: {result.get('error', 'Unknown')}")
    
    # Compare results
    if backtest_results:
        print(f"\n  {BOLD}📊 Model Comparison:{RESET}")
        print(f"  {'Model':<15} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Trades':>8}")
        print(f"  {'-' * 55}")
        
        best_model = None
        best_sharpe = -999
        
        for br in backtest_results:
            m = br["model"]
            bt = m.get("backtest", {})
            name = m.get("name", m["model_type"])[:15]
            sharpe = bt.get("sharpe", 0)
            ret = bt.get("total_return", 0)
            dd = bt.get("max_drawdown", 0)
            trades = bt.get("n_trades", 0)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model = m
            
            print(f"  {name:<15} {sharpe:>8.2f} {ret:>9.1f}% {dd:>9.1f}% {trades:>8}")
    
    # =========================================
    # Step 4: Promote Best Model
    # =========================================
    print_step(4, "Promote Best Model to Production")
    
    if best_model:
        model_id = best_model["model_id"]
        print(f"  Best model: {best_model.get('name', best_model['model_type'])} (Sharpe: {best_sharpe:.2f})")
        
        result = api_call(base_url, "POST", f"/models/{model_id}/promote")
        
        if result.get("promoted_id"):
            print_ok(f"Promoted: {result['promoted_id'][:16]}...")
            
            # Verify promotion
            promoted = api_call(base_url, "GET", "/models/promoted")
            if promoted.get("promoted_id"):
                print_ok(f"Production model active: {promoted['model_type']}")
        else:
            print_warn(f"Promotion failed: {result}")
    
    # =========================================
    # Step 5: Predict with Production Model
    # =========================================
    print_step(5, "Predict with Production Model")
    
    for ticker in ["AAPL", "GOOGL", "NVDA"]:
        result = api_call(base_url, "POST", "/predict", {
            "ticker": ticker,
            "horizons": [5],
        })
        
        if result.get("success"):
            pred = result.get("prediction", "?")
            prob = result.get("probability", {})
            signal = result.get("signal", "?")
            conf = result.get("confidence", 0)
            
            emoji = "📈" if pred == 1 else "📉"
            print(f"  {emoji} {ticker}: {signal} (up: {prob.get('up', 0):.1%}, confidence: {conf:.1%})")
        else:
            print_warn(f"{ticker}: {result.get('error', 'Failed')}")
    
    # =========================================
    # Step 6: Technical Analysis Agent
    # =========================================
    print_step(6, "Technical Analysis Agent")
    
    result = api_call(base_url, "POST", "/agents/technical", {
        "ticker": "AAPL",
        "include_shap": True,
        "top_features": 3,
    })
    
    if result.get("success"):
        print_ok(f"Prediction: {result.get('prediction')} ({result.get('probability', 0):.1%})")
        print(f"\n  {BOLD}Summary:{RESET}")
        print(f"  {result.get('summary', 'N/A')}")
        
        print(f"\n  {BOLD}Top Features:{RESET}")
        for feat in result.get("top_features", [])[:3]:
            direction = "↑" if feat["direction"] == "bullish" else "↓"
            print(f"  • {feat['name']}: {feat['value']:.4f} {direction} (SHAP: {feat['contribution']:.4f})")
        
        print(f"\n  {BOLD}Signals:{RESET}")
        for sig in result.get("signals", [])[:2]:
            emoji = "🟢" if sig["signal"] == "bullish" else "🔴"
            print(f"  {emoji} {sig['indicator']}: {sig['description']}")
    else:
        print_warn(f"Agent failed: {result.get('error', 'Unknown')}")
    
    # =========================================
    # Step 7: RAG - Why did the model predict this?
    # =========================================
    print_step(7, "RAG: 'Why does the model predict this?'")
    
    # First, refresh the index
    api_call(base_url, "POST", "/rag/index/refresh")
    
    questions = [
        "What models have been trained?",
        "Why did the model predict AAPL will go up?",
        "What features are most important for predictions?",
    ]
    
    for q in questions:
        print(f"\n  {BOLD}Q: {q}{RESET}")
        
        result = api_call(base_url, "POST", "/rag/answer", {
            "query": q,
            "top_k": 3,
        })
        
        if result.get("answer"):
            print(f"  A: {result['answer'][:200]}...")
            print(f"  {DIM}(confidence: {result.get('confidence', 0):.2f}, evidence: {len(result.get('evidence', []))} docs){RESET}")
        else:
            print_warn("No answer generated")
    
    # =========================================
    # Summary
    # =========================================
    elapsed = time.time() - start_time
    
    print_header("✨ Demo Complete")
    
    print(f"  {BOLD}What we demonstrated:{RESET}")
    print(f"  1. ✓ Trained {len(trained_models)} models (Logistic, XGBoost, LightGBM)")
    print(f"  2. ✓ Compared backtest performance with transaction costs & slippage")
    print(f"  3. ✓ Promoted best model to production")
    print(f"  4. ✓ Made predictions using production model")
    print(f"  5. ✓ Generated technical analysis with SHAP explanations")
    print(f"  6. ✓ Answered questions using RAG")
    
    print(f"\n  {BOLD}Key V3 Features Shown:{RESET}")
    print(f"  • Async training with job queue")
    print(f"  • Experiment tracking (git SHA, data hash)")
    print(f"  • 5 model types with unified interface")
    print(f"  • Hyperparameter search (Optuna)")
    print(f"  • Model cache + production promotion")
    print(f"  • FAISS-based RAG for explanations")
    print(f"  • Position sizing & portfolio backtesting")
    print(f"  • Technical analysis agents")
    
    print(f"\n  {GREEN}Completed in {elapsed:.1f}s{RESET}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quant AI V3 Interview Demo")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip training, use existing models",
    )
    args = parser.parse_args()
    
    success = demo(args.base_url, args.quick)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
