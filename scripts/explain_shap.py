from app.explain.shap_explainer import ShapExplainer
from app.explain.shap_to_text import shap_summary_to_text
from app.vector.embedder import embed_texts
from app.vector.index import FaissIndex


def main():
    ticker = "AAPL"
    lookback = 1500
    top_k = 8

    # === 1. SHAP explain ===
    explainer = ShapExplainer("artifacts/model.joblib")

    result = explainer.explain(
        ticker=ticker,
        lookback=lookback,
        top_k=top_k,
    )

    print("SHAP result:")
    print(result)

    # === 2. Convert SHAP → text ===
    text = shap_summary_to_text(result)

    # === 3. Embed ===
    vec = embed_texts([text])

    # === 4. Vector DB (v1: in-memory FAISS) ===
    index = FaissIndex(dim=vec.shape[1])

    index.add(
        vec,
        [
            {
                "type": "shap_summary",
                "ticker": ticker,
                "text": text,
            }
        ],
    )

    print("SHAP summary indexed into vector DB.")


if __name__ == "__main__":
    main()
