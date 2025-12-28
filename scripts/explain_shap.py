from app.explain.shap_explainer import ShapExplainer

explainer = ShapExplainer("artifacts/model.joblib")

result = explainer.explain(
    ticker="AAPL",
    lookback=1500,
    top_k=8,
)

print(result)
