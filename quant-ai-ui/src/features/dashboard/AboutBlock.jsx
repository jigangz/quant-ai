export function AboutBlock({ ticker, name, industry = "科技", modelMeta }) {
  const companyName = name ?? ticker;
  const sentence = modelMeta
    ? `${companyName} 是一家 ${industry} 公司。AI 模型基于过去 2 年日线数据训练，使用技术指标（RSI/MACD/Bollinger）、动量、波动率、成交量、情绪、新闻 6 组特征。当前使用 ${modelMeta.model_type ?? "—"} · run #${modelMeta.training_run_id ?? "—"} · git ${modelMeta.git_sha?.slice(0, 7) ?? "—"}（${modelMeta.trained_on ?? "—"} 训练，AUC ${modelMeta.metrics?.val_auc?.toFixed(2) ?? "—"}）。`
    : `${companyName} 是一家 ${industry} 公司。AI 模型信息加载中...`;
  return (
    <p className="text-[10.5px] text-muted leading-relaxed mb-4">{sentence}</p>
  );
}
