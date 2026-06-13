"""Factor research — IC ranking → correlation de-dup → selection (V5 Phase B).

Wide net (~35 candidate factors) → per-date Rank IC + IR → correlation
matrix → drop redundant (|corr| > 0.9, keep the higher |IC|) → drop weak
(|IC| below floor) → a lean, orthogonal, effective set. The selected set is
what Phase C's ranking model trains on (saved to config).

Usage: python -m scripts.factor_research [--tickers N] [--ic-floor 0.01]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.ml.features.cross_section import cross_section_normalize
from scripts.xs_eval import EXCLUDE, add_xs_label, load_panel

OUT = Path(__file__).resolve().parent.parent / "config" / "xs_selected_factors.json"


def rank_ic(panel: pd.DataFrame, factor: str, ret: str = "fwd_ret") -> tuple[float, float]:
    ics = []
    for _, g in panel.groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g[factor], g[ret])
        if ic == ic:
            ics.append(ic)
    arr = np.array(ics)
    return float(arr.mean()), float(arr.mean() / (arr.std() + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top", type=float, default=0.30)
    ap.add_argument("--ic-floor", type=float, default=0.01, help="drop |IC| below this")
    ap.add_argument("--corr-max", type=float, default=0.90, help="drop |corr| above this")
    args = ap.parse_args()

    print(f"Loading panel ({args.tickers} tickers)...")
    panel, _ = load_panel(args.tickers)
    panel = add_xs_label(panel, args.horizon, args.top)
    factors = [c for c in panel.columns if c not in EXCLUDE]
    panel = cross_section_normalize(panel, factors).dropna(subset=factors)
    print(f"  {len(factors)} candidate factors, {len(panel):,} rows\n")

    # 1) Rank IC per factor (spearman is invariant to the z-score, so this is
    #    the factor's raw predictive power)
    rows = []
    for f in factors:
        ic, ir = rank_ic(panel, f)
        rows.append({"factor": f, "ic": ic, "abs_ic": abs(ic), "ir": ir})
    ic_df = pd.DataFrame(rows).sort_values("abs_ic", ascending=False).reset_index(drop=True)
    ic_map = dict(zip(ic_df["factor"], ic_df["abs_ic"]))

    print("=== Factor Rank IC (sorted) ===")
    for _, r in ic_df.iterrows():
        flag = "" if r.abs_ic >= args.ic_floor else "  (weak)"
        print(f"  {r.factor:18} IC {r.ic:+.4f}  IR {r.ir:+.2f}{flag}")

    # 2) Correlation matrix
    corr = panel[factors].corr().abs()

    # 3) De-dup: walk factors high-IC first; drop one that's >corr_max with an
    #    already-kept factor (keep the higher-IC of each redundant pair)
    kept, dropped_corr = [], []
    for f in ic_df["factor"]:
        if any(corr.loc[f, k] > args.corr_max for k in kept):
            dropped_corr.append(f)
        else:
            kept.append(f)

    # 4) Drop weak (low |IC|)
    selected = [f for f in kept if ic_map[f] >= args.ic_floor]
    dropped_weak = [f for f in kept if ic_map[f] < args.ic_floor]

    print(f"\n=== Selection ({len(factors)} → {len(selected)}) ===")
    print(f"  dropped (redundant, |corr|>{args.corr_max}): {dropped_corr}")
    print(f"  dropped (weak, |IC|<{args.ic_floor}): {dropped_weak}")
    print(f"  SELECTED ({len(selected)}): {selected}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"selected_factors": selected, "ic": ic_map}, indent=2))
    print(f"\n  saved → {OUT}")
    return 0


if __name__ == "__main__":
    main()
