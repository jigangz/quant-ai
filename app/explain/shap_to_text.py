def shap_summary_to_text(summary: dict) -> str:
    ticker = summary["ticker"]
    features = summary["top_features"]

    top_feats = [f["feature"] for f in features[:3]]
    weak_feats = [f["feature"] for f in features[-2:]]

    text = (
        f"Ticker {ticker}: "
        f"Top drivers were {', '.join(top_feats)}. "
        f"{top_feats[0]} dominated predictions. "
        f"{', '.join(weak_feats)} had low impact."
    )

    return text
