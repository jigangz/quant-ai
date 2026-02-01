#!/usr/bin/env python3
"""
Demo 30s - 快速演示

展示:
1. Health check - 服务状态
2. 列出已有模型

用法:
    python scripts/demo_30s.py
    python scripts/demo_30s.py --base-url http://your-server:8000
"""

import argparse
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


def demo(base_url: str):
    """Run the 30-second demo."""
    
    print(f"\n{BOLD}🚀 Quant AI - 30 Second Demo{RESET}")
    print(f"Base URL: {base_url}\n")
    
    start = time.time()
    
    # Step 1: Health Check
    print_step(1, "Health Check")
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        print_ok(f"Status: {data.get('status', 'unknown')}")
        
        settings = data.get("settings", {})
        if settings:
            print(f"   • Environment: {settings.get('env', 'N/A')}")
            print(f"   • Default model: {settings.get('default_model_type', 'N/A')}")
            print(f"   • Feature groups: {settings.get('default_feature_groups', [])}")
            print(f"   • Providers: {settings.get('providers_enabled', [])}")
    
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to {base_url}")
        print("   Make sure the server is running:")
        print("   $ docker-compose up")
        print("   or")
        print("   $ uvicorn app.main:app --reload")
        return False
    
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False
    
    # Step 2: List Models
    print_step(2, "List Models")
    
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        
        if resp.status_code == 404:
            print_warn("No /models endpoint yet (V1 mode)")
            print("   Train a model first with POST /train")
        else:
            resp.raise_for_status()
            models = resp.json()
            
            if isinstance(models, list) and len(models) > 0:
                print_ok(f"Found {len(models)} model(s):")
                for m in models[:5]:  # Show first 5
                    name = m.get("name", m.get("id", "unknown"))
                    model_type = m.get("model_type", "?")
                    tickers = m.get("tickers", [])
                    metrics = m.get("metrics", {})
                    auc = metrics.get("auc", "N/A")
                    print(f"   • {name} ({model_type}) - {tickers} - AUC: {auc}")
            else:
                print_warn("No models found")
                print("   Train one with: POST /train")
    
    except Exception as e:
        print_warn(f"Could not list models: {e}")
    
    # Summary
    elapsed = time.time() - start
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{GREEN}✓ Demo completed in {elapsed:.1f}s{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    print(f"\n{BOLD}Next steps:{RESET}")
    print("  • Run full demo: python scripts/demo_2min.py")
    print("  • Train a model: POST /train with tickers, model_type")
    print("  • Run backtest:  POST /backtest with model_id")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quant AI 30-second demo")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    
    success = demo(args.base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
