import { Gauge } from "./Gauge";

const MOMENTUM_INDICATORS = ["RSI", "MACD", "Stochastic"];
const MA_INDICATORS = ["MA", "SMA", "EMA", "MA_CROSS"];

const DIR_VAL = { bullish: 1, bearish: -1, neutral: 0, up: 1, down: -1 };
const STR_VAL = { strong: 1.0, moderate: 0.67, weak: 0.33, "": 0.5 };

function aggregateSignals(signals, matchers) {
  if (!signals || signals.length === 0) return 0;
  const matched = signals.filter((s) =>
    matchers.some((m) => (s.indicator ?? "").toUpperCase().includes(m.toUpperCase()))
  );
  if (matched.length === 0) return 0;
  const sum = matched.reduce((acc, s) => {
    const dv = DIR_VAL[s.signal ?? s.direction] ?? 0;
    const sv = STR_VAL[(s.strength ?? "").toLowerCase()] ?? 0.5;
    return acc + dv * sv;
  }, 0);
  const avg = sum / matched.length;
  return Math.max(-2, Math.min(2, avg * 2));
}

function scoreToLabel(score) {
  if (score >= 1.5) return "强烈买入";
  if (score >= 0.5) return "买入";
  if (score <= -1.5) return "强烈卖出";
  if (score <= -0.5) return "卖出";
  return "中立";
}

function aiScore(prediction, probability, confidence) {
  if (probability?.up == null) return 0;
  const p = probability.up;
  let base;
  if (p < 0.3) base = -2;
  else if (p < 0.45) base = -1;
  else if (p <= 0.55) base = 0;
  else if (p <= 0.7) base = 1;
  else base = 2;
  const mult = confidence === "high" ? 1.0 : confidence === "medium" ? 0.7 : 0.4;
  return Math.max(-2, Math.min(2, base * mult));
}

export function GaugesSection({ prediction, probability, confidence, signals = [] }) {
  const oscScore = aggregateSignals(signals, MOMENTUM_INDICATORS);
  const maScore = aggregateSignals(signals, MA_INDICATORS);
  const ai = aiScore(prediction, probability, confidence);

  return (
    <section className="mb-4">
      <h3 className="text-sm font-bold text-foreground">技术指标 ›</h3>
      <p className="text-[10px] text-muted mb-2">总结指标的建议</p>
      <div className="grid grid-cols-3 gap-4">
        <Gauge label="震荡指标 (RSI/MACD)" score={oscScore} scoreLabel={scoreToLabel(oscScore)} />
        <Gauge label="🤖 AI 模型总结" score={ai} scoreLabel={scoreToLabel(ai)} emphasized />
        <Gauge label="移动平均线" score={maScore} scoreLabel={scoreToLabel(maScore)} />
      </div>
    </section>
  );
}
