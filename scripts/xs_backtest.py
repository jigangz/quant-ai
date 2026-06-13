"""Backtest the xs_strong Top-N ranking portfolio out-of-sample — V5 Phase E.

Loads the published xs_strong model (DB blob), builds the OOS panel (dates after
the model's train_end_date), and reports the Top-N selection strategy net of
transaction costs vs an equal-weight-universe benchmark — long-only and
long-short, at the top decile and top-30%, gross and net.

Read-only (no DB writes).

Usage:
    python -m scripts.xs_backtest [--cost-bps 10] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)
logging.disable(logging.WARNING)

from app.backtest.xs_portfolio import backtest_xs_portfolio, build_backtest_panel  # noqa: E402
from app.db.prices_repo import get_prices_batch  # noqa: E402
from app.ml.models.xgboost_model import XGBoostModel  # noqa: E402

H = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=int, default=10, help="per-side transaction cost (bps)")
    ap.add_argument("--limit", type=int, default=None, help="cap the universe size")
    ap.add_argument("--since", default=None,
                    help="OOS cutoff override (YYYY-MM-DD). Default = model.test_start_date "
                         "(honest val/test boundary), else train_end_date.")
    args = ap.parse_args()

    eng = create_engine(os.environ["DATABASE_URL"])
    with eng.connect() as c:
        mid = c.execute(
            text("SELECT id FROM model_registry WHERE label_type='xs_strong' "
                 "ORDER BY created_at DESC LIMIT 1")
        ).scalar()
        if not mid:
            print("No xs_strong model in the registry. Publish one first.")
            return 1
        row = c.execute(
            text("SELECT blob FROM model_artifacts WHERE model_id=:i"), {"i": mid}
        ).fetchone()
    if not row:
        print(f"No blob for {mid}.")
        return 1

    model = XGBoostModel.from_zip_bytes(bytes(row[0]))
    universe = list(model.metadata.tickers)
    if args.limit:
        universe = universe[: args.limit]

    cutoff = args.since or getattr(model.metadata, "test_start_date", None) \
        or getattr(model.metadata, "train_end_date", None)
    print(f"Building OOS panel for {len(universe)} tickers (reading prices DB)...")
    prices = get_prices_batch(universe)
    panel, factor_cols = build_backtest_panel(model, prices, H=H, since=args.since)
    if panel.empty:
        print("Empty OOS panel — check the OOS cutoff / prices coverage.")
        return 1
    print(f"OOS cutoff: {cutoff}  (test-split boundary if available; --since overrides)")

    dates = sorted(pd.Timestamp(d) for d in panel["date"].unique())
    n_reb = len(dates[::H])
    print(
        f"model={mid}\n"
        f"OOS window: {dates[0].date()} .. {dates[-1].date()}  "
        f"({len(dates)} trading days, ~{n_reb} rebalances at H={H}, cost={args.cost_bps}bps/side)\n"
    )

    hdr = f"{'strategy':<26}{'netCAGR%':>9}{'netSharpe':>10}{'grossSharpe':>12}{'MaxDD%':>8}{'turn%':>7}"
    print(hdr)
    print("-" * len(hdr))

    # Benchmark (equal-weight universe) — same for every cell.
    base = backtest_xs_portfolio(panel, model, factor_cols=factor_cols, top_pct=0.10,
                                 H=H, cost_bps=args.cost_bps)
    bm = base["benchmark"]["metrics"]
    print(f"{'Benchmark (EW universe)':<26}{bm['cagr']:>9}{bm['sharpe']:>10}{'-':>12}{bm['max_drawdown']:>8}{'-':>7}")

    for cname, ls in [("Long-only", False), ("Long-short", True)]:
        for pname, pct in [("decile", 0.10), ("top-30%", 0.30)]:
            bt = backtest_xs_portfolio(panel, model, factor_cols=factor_cols,
                                       top_pct=pct, H=H, cost_bps=args.cost_bps, long_short=ls)
            net, gross = bt["net"]["metrics"], bt["gross"]["metrics"]
            print(
                f"{f'{cname} {pname}':<26}{net['cagr']:>9}{net['sharpe']:>10}"
                f"{gross['sharpe']:>12}{net['max_drawdown']:>8}{bt['avg_turnover']*100:>6.1f}%"
            )

    print("\nBenchmark = equal-weight whole universe (isolates selection skill).")
    print("Honest read: compare gross vs net Sharpe — how much of the edge survives costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
