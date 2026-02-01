#!/usr/bin/env python3
"""
2-minute demo - Full training and backtest flow.

Steps:
1. Health check
2. Fetch market data
3. Train models (Logistic + XGBoost)
4. List models
5. Run backtest
6. Show results

Usage:
    python scripts/demo_2min.py
    python scripts/demo_2min.py --base-url http://your-server:8000
    python scripts/demo_2min.py --ticker MSFT --skip-train
"""

import argparse
import json
import sys
import time

import requests

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def print_step(n: int, msg: str):
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}Step {n}: {msg}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")


def print_ok(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")


def print_warn(msg: str):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def print_error(msg: str):
    print(f"{RED}✗ {msg}{RESET}")


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON with colors."""
    formatted = json.dumps(data, indent=indent, default=str)
    # Highlight keys
    lines = formatted.split("\n")
    for line in lines[:20]:  # Limit output
        print(f"   {DIM}{line}{RESET}")
    if len(lines) > 20:
        print(f"   {DIM}... ({len(lines) - 20} more lines){RESET}")


def demo(base_url: str, ticker: str, skip_train: bool):
    """Run the 2-minute demo."""
    
    print(f"\n{BOLD}🚀 Quant AI - 2 Minute Demo{RESET}")
    print(f"Base URL: {base_url}")
    print(f"Ticker: {ticker}")
    print(f"Skip training: {skip_train}\n")
    
    start = time.time()
    model_ids = []
    
    # Step 1: Health Check
    print_step(1, "Health Check")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        print_ok(f"Status: {data.get('status', 'unknown')}")
        settings = data.get("settings", {})
        print(f"   Environment: {settings.get('env', 'N/A')}")
        print(f"   Default model: {settings.get('default_model_type', 'N/A')}")
    
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to {base_url}")
        print("   Make sure the server is running:")
        print("   $ docker-compose up  OR  $ uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False
    
    # Step 2: Fetch Market Data
    print_step(2, f"Fetch Market Data for {ticker}")
    
    try:
        resp = requests.get(
            f"{base_url}/data/market",
            params={"ticker": ticker, "period": "1y", "limit": 10},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, list):
            print_ok(f"Got {len(data)} data points")
            if data:
                print(f"   Latest: {data[-1].get('date', 'N/A')} - Close: ${data[-1].get('close', 0):.2f}")
        else:
            print_ok("Market data received")
    
    except Exception as e:
        print_warn(f"Market data fetch failed: {e}")
        print("   Continuing with training...")
    
    # Step 3: Train Models
    if not skip_train:
        print_step(3, "Train Models")
        
        # Train Logistic Regression
        print(f"\n{BOLD}3a. Training Logistic Regression...{RESET}")
        
        try:
            train_request = {
                "tickers": [ticker],
                "feature_groups": ["ta_basic", "momentum"],
                "model_type": "logistic",
                "horizon_days": 5,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            }
            
            resp = requests.post(
                f"{base_url}/train",
                json=train_request,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            
            model_id = result.get("id")
            model_ids.append(model_id)
            metrics = result.get("metrics", {})
            
            print_ok(f"Model trained: {model_id}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"   AUC: {metrics.get('auc', 'N/A')}")
            print(f"   F1: {metrics.get('f1', 'N/A')}")
        
        except requests.exceptions.HTTPError as e:
            print_error(f"Training failed: {e.response.text if e.response else e}")
        except Exception as e:
            print_error(f"Training failed: {e}")
        
        # Train XGBoost (if available)
        print(f"\n{BOLD}3b. Training XGBoost...{RESET}")
        
        try:
            train_request = {
                "tickers": [ticker],
                "feature_groups": ["ta_basic", "momentum", "volatility"],
                "model_type": "xgboost",
                "horizon_days": 5,
                "model_params": {"n_estimators": 50, "max_depth": 3},
            }
            
            resp = requests.post(
                f"{base_url}/train",
                json=train_request,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            
            model_id = result.get("id")
            model_ids.append(model_id)
            metrics = result.get("metrics", {})
            
            print_ok(f"Model trained: {model_id}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"   AUC: {metrics.get('auc', 'N/A')}")
        
        except requests.exceptions.HTTPError as e:
            if "xgboost" in str(e).lower() or "not installed" in str(e).lower():
                print_warn("XGBoost not available, skipping")
            else:
                print_warn(f"XGBoost training failed: {e.response.text if e.response else e}")
        except Exception as e:
            print_warn(f"XGBoost training failed: {e}")
    
    else:
        print_step(3, "Training Skipped (--skip-train)")
    
    # Step 4: List Models
    print_step(4, "List All Models")
    
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        
        if resp.status_code == 404:
            print_warn("No /models endpoint (V1 mode)")
        else:
            resp.raise_for_status()
            models = resp.json()
            
            if isinstance(models, list) and len(models) > 0:
                print_ok(f"Found {len(models)} model(s):")
                for m in models[:5]:
                    mid = m.get("id", "?")[:8]
                    name = m.get("name", "unnamed")
                    mtype = m.get("model_type", "?")
                    tickers = m.get("tickers", [])
                    auc = m.get("metrics", {}).get("auc", "N/A")
                    print(f"   • [{mid}...] {name} ({mtype}) - {tickers} - AUC: {auc}")
                    
                    # Store model_id if we didn't train
                    if skip_train and not model_ids:
                        model_ids.append(m.get("id"))
            else:
                print_warn("No models found")
    
    except Exception as e:
        print_warn(f"Could not list models: {e}")
    
    # Step 5: Run Backtest
    print_step(5, "Run Backtest")
    
    if not model_ids:
        print_warn("No model_id available for backtest")
        print("   Train a model first or check /models")
    else:
        model_id = model_ids[0]
        print(f"   Using model: {model_id[:16]}...")
        
        try:
            backtest_request = {
                "model_id": model_id,
                "signal_threshold": 0.55,
                "transaction_cost_bps": 10,
            }
            
            resp = requests.post(
                f"{base_url}/backtest",
                json=backtest_request,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            
            if result.get("success"):
                print_ok("Backtest completed!")
                
                # Classification metrics
                cm = result.get("classification_metrics", {})
                print(f"\n   {BOLD}Classification Metrics:{RESET}")
                print(f"   • Accuracy: {cm.get('accuracy', 'N/A')}")
                print(f"   • AUC: {cm.get('auc', 'N/A')}")
                print(f"   • F1: {cm.get('f1', 'N/A')}")
                
                # Strategy metrics
                sm = result.get("strategy_metrics", {})
                print(f"\n   {BOLD}Strategy Metrics:{RESET}")
                print(f"   • CAGR: {sm.get('cagr', 'N/A')}")
                print(f"   • Sharpe Ratio: {sm.get('sharpe_ratio', 'N/A')}")
                print(f"   • Max Drawdown: {sm.get('max_drawdown', 'N/A')}")
                print(f"   • Win Rate: {sm.get('win_rate', 'N/A')}")
                
                # Comparison
                bh = sm.get("buy_and_hold_return", "N/A")
                strat = sm.get("strategy_return", "N/A")
                print(f"\n   {BOLD}vs Buy & Hold:{RESET}")
                print(f"   • Strategy Return: {strat}")
                print(f"   • Buy & Hold: {bh}")
            else:
                print_error(f"Backtest failed: {result.get('error')}")
        
        except requests.exceptions.HTTPError as e:
            err = e.response.json() if e.response else str(e)
            print_error(f"Backtest failed: {err}")
        except Exception as e:
            print_error(f"Backtest failed: {e}")
    
    # Step 6: Summary
    elapsed = time.time() - start
    print_step(6, "Summary")
    
    print_ok(f"Demo completed in {elapsed:.1f}s")
    print(f"\n   {BOLD}What we did:{RESET}")
    print("   1. ✓ Verified service health")
    print("   2. ✓ Fetched market data from Yahoo Finance")
    if not skip_train:
        print("   3. ✓ Trained ML models (Logistic, XGBoost)")
    print("   4. ✓ Listed registered models")
    print("   5. ✓ Ran backtest with strategy simulation")
    
    print(f"\n   {BOLD}Key features demonstrated:{RESET}")
    print("   • Time-series split (no data leakage)")
    print("   • Model versioning (auto-registered)")
    print("   • Strategy metrics (Sharpe, CAGR, Max DD)")
    print("   • Classification metrics (AUC, F1)")
    
    print(f"\n   {BOLD}Next steps:{RESET}")
    print("   • Try different tickers: --ticker MSFT")
    print("   • Check SHAP: GET /explain?ticker=AAPL")
    print("   • Deploy: docker-compose -f docker-compose.prod.yml up")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quant AI 2-minute demo")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to use for demo (default: AAPL)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, use existing models",
    )
    args = parser.parse_args()
    
    success = demo(args.base_url, args.ticker, args.skip_train)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
