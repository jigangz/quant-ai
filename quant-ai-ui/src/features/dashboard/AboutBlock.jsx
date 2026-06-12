export function AboutBlock({ ticker, name, industry = "Technology", modelMeta }) {
  const companyName = name ?? ticker;
  const sentence = modelMeta
    ? `${companyName} is a ${industry} company. The AI model is trained on the past 2 years of daily data across 6 feature groups: technical indicators (RSI/MACD/Bollinger), momentum, volatility, volume, sentiment, and news. Currently using ${modelMeta.model_type ?? "—"} · run #${modelMeta.training_run_id ?? "—"} · git ${modelMeta.git_sha?.slice(0, 7) ?? "—"} (trained ${modelMeta.trained_on ?? "—"}, AUC ${modelMeta.metrics?.val_auc?.toFixed(2) ?? "—"}).`
    : `${companyName} is a ${industry} company. Loading AI model details...`;
  return (
    <p className="text-[10.5px] text-muted leading-relaxed mb-4">{sentence}</p>
  );
}
