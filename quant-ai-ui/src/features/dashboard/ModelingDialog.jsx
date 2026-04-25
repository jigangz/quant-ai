import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { usePromoteModel, useTrain } from "@/api/queries";
import { useModelsForTickerByLabelType } from "@/api/queries";

// V4 Pivot · Phase 2 · Sub 4 Modeling Dialog (MVP)
//
// TradingView-style "indicators & strategies" modal, slimmed for MVP:
// - Top segmented control: 预测目标 (direction / volatility)
// - Three tabs: 我的模型 (per-ticker list) · 预设模板 · 自定义训练
// - Promote flow: 点"使用此模型" -> /models/{id}/promote -> dialog closes
// - Custom train flow: 点"训练" -> /train -> 训完自动 promote + 关
//
// Deferred (polish for post-MVP): Optuna control, progress-bar streaming,
// preset library expansion, model deletion, search box.

// /train (legacy direction/volatility) accepts all 5 model types. Use the
// stronger XGBoost as default preset since prod has libgomp1 + xgboost wheel.
const PRESETS = {
  direction: [
    { id: "xgb_dir_default", label: "⚡ 快速 XGBoost", model_type: "xgboost", horizon_days: 5 },
    { id: "logistic_dir", label: "🎯 稳健 Logistic", model_type: "logistic", horizon_days: 5 },
  ],
  volatility: [
    { id: "xgb_vol_default", label: "⚡ 快速 XGBoost", model_type: "xgboost", horizon_days: 5 },
    { id: "rf_vol_default", label: "🌲 稳健 RandomForest", model_type: "random_forest", horizon_days: 5 },
  ],
};

const LABEL_TYPES = [
  { value: "direction", label: "方向 (baseline)" },
  { value: "volatility", label: "波动率 (V4 P2)" },
  // meta_label + regime deferred to V4 P3+
];

export function ModelingDialog({ ticker, children }) {
  const [open, setOpen] = useState(false);
  const [labelType, setLabelType] = useState("direction");
  const [tab, setTab] = useState("mine");
  const [customModelType, setCustomModelType] = useState("xgboost");
  const [customHorizon, setCustomHorizon] = useState(5);

  const modelsQ = useModelsForTickerByLabelType(ticker, labelType);
  const trainM = useTrain();
  const promoteM = usePromoteModel();
  const qc = useQueryClient();

  const handlePromote = async (modelId) => {
    await promoteM.mutateAsync(modelId);
    qc.invalidateQueries({ queryKey: ["agent", "technical", ticker] });
    qc.invalidateQueries({ queryKey: ["predict-volatility", ticker] });
    qc.invalidateQueries({ queryKey: ["models", "byTicker+label", ticker] });
    setOpen(false);
  };

  const handleTrain = async (payload) => {
    const res = await trainM.mutateAsync(payload);
    // Async mode returns { run_id }; sync returns { id }.
    const modelId = res.id || res.model_id;
    if (modelId) {
      await handlePromote(modelId);
    } else {
      setOpen(false);  // async mode: dialog closes; runs page tracks progress
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>🧪 建模 · {ticker}</DialogTitle>
          <DialogDescription>
            选择 <b>预测目标</b> 和建模方式。V4 Pivot 支持多目标(方向 / 波动率)。
          </DialogDescription>
        </DialogHeader>

        {/* Top label_type toggle */}
        <div className="flex gap-2 p-1 bg-surface-muted rounded-md">
          {LABEL_TYPES.map((lt) => (
            <button
              key={lt.value}
              type="button"
              onClick={() => setLabelType(lt.value)}
              className={`flex-1 text-xs py-1.5 rounded transition-colors ${
                labelType === lt.value
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted hover:bg-surface"
              }`}
            >
              {lt.label}
            </button>
          ))}
        </div>

        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="grid grid-cols-3">
            <TabsTrigger value="mine">我的模型</TabsTrigger>
            <TabsTrigger value="preset">预设模板</TabsTrigger>
            <TabsTrigger value="custom">自定义训练</TabsTrigger>
          </TabsList>

          {/* Mine · list of models filtered by label_type via G3 backend */}
          <TabsContent value="mine" className="min-h-[240px]">
            {modelsQ.isLoading ? (
              <p className="text-sm text-muted py-4">加载中…</p>
            ) : (modelsQ.data ?? []).length === 0 ? (
              <EmptyMine labelType={labelType} />
            ) : (
              <ul className="divide-y divide-border text-sm">
                {modelsQ.data.map((m) => (
                  <ModelRow key={m.id} model={m} onUse={() => handlePromote(m.id)} />
                ))}
              </ul>
            )}
          </TabsContent>

          {/* Preset · fixed cards */}
          <TabsContent value="preset" className="min-h-[240px]">
            <div className="grid grid-cols-2 gap-3">
              {(PRESETS[labelType] || []).map((p) => (
                <PresetCard
                  key={p.id}
                  preset={p}
                  disabled={trainM.isPending}
                  onUse={() =>
                    handleTrain({
                      tickers: [ticker],
                      model_type: p.model_type,
                      feature_groups: ["ta_basic"],
                      horizon_days: p.horizon_days,
                      label_type: labelType,
                      save_model: true,
                    })
                  }
                />
              ))}
            </div>
          </TabsContent>

          {/* Custom · minimal form */}
          <TabsContent value="custom" className="min-h-[240px] space-y-3">
            <div>
              <Label>算法</Label>
              <select
                value={customModelType}
                onChange={(e) => setCustomModelType(e.target.value)}
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="logistic">logistic</option>
                <option value="random_forest">random_forest</option>
                <option value="xgboost">xgboost</option>
                <option value="lightgbm">lightgbm</option>
                <option value="catboost">catboost</option>
              </select>
            </div>
            <div>
              <Label>预测 horizon (天)</Label>
              <Input
                type="number"
                min={1}
                max={60}
                value={customHorizon}
                onChange={(e) => setCustomHorizon(Number(e.target.value))}
              />
            </div>
            <Button
              disabled={trainM.isPending}
              onClick={() =>
                handleTrain({
                  tickers: [ticker],
                  model_type: customModelType,
                  feature_groups: ["ta_basic"],
                  horizon_days: customHorizon,
                  label_type: labelType,
                  save_model: true,
                })
              }
            >
              {trainM.isPending ? "训练中…" : `训练 ${customModelType} · ${labelType}`}
            </Button>
          </TabsContent>
        </Tabs>

        {trainM.isError && (
          <p className="text-xs text-down">训练失败: {String(trainM.error?.message ?? trainM.error)}</p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ModelRow({ model, onUse }) {
  const metrics = model.metrics ?? {};
  const perf =
    metrics.val_auc != null
      ? `AUC ${Number(metrics.val_auc).toFixed(3)}`
      : metrics.val_mae != null
        ? `MAE ${Number(metrics.val_mae).toFixed(4)}`
        : "—";
  return (
    <li className="flex items-center justify-between py-2.5">
      <div className="min-w-0">
        <div className="font-medium truncate">{model.name}</div>
        <div className="text-[11px] text-muted">
          {model.model_type} · {perf} · h={model.horizon_days ?? 5}
        </div>
      </div>
      <Button size="sm" onClick={onUse}>
        使用此模型
      </Button>
    </li>
  );
}

function PresetCard({ preset, disabled, onUse }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onUse}
      className="text-left border border-border rounded-lg p-3 hover:bg-surface-muted disabled:opacity-50 transition-colors"
    >
      <div className="text-sm font-semibold">{preset.label}</div>
      <div className="text-[11px] text-muted mt-1">
        {preset.model_type} · horizon {preset.horizon_days}d · 默认 ta_basic
      </div>
      <div className="text-[11px] text-accent mt-2">点击训练 →</div>
    </button>
  );
}

function EmptyMine({ labelType }) {
  return (
    <div className="text-xs text-muted py-6 text-center border border-dashed border-border rounded">
      还没有 label_type=<b>{labelType}</b> 的模型。
      <br />
      去 <b>预设模板</b> 或 <b>自定义训练</b> 标签训练一个。
    </div>
  );
}
