import { Link } from "react-router-dom";
import { ModelingDialog } from "./ModelingDialog";

export function CTARow({ ticker, prediction }) {
  const side = prediction === 1 ? "buy" : prediction === 0 ? "sell" : "buy";
  return (
    <div className="grid grid-cols-2 gap-2.5 mt-4">
      <Link
        to={`/trading?ticker=${ticker}&side=${side}&suggestion_source=dashboard`}
        className="bg-accent text-accent-foreground text-sm font-bold py-2.5 rounded-md text-center hover:bg-accent-hover transition-colors"
      >
        🛒 基于此信号纸上下单
      </Link>
      <ModelingDialog ticker={ticker}>
        <button
          type="button"
          className="bg-surface border border-surface-border text-foreground text-sm py-2.5 rounded-md text-center hover:bg-surface-muted transition-colors w-full"
        >
          🧪 选模型 / 训练 · 多目标 (V4 P2)
        </button>
      </ModelingDialog>
    </div>
  );
}
