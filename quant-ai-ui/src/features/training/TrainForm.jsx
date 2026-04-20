import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select";
import ErrorState from "../../components/ErrorState";
import { useTrain, useModelTypes } from "../../api/queries";

const schema = z.object({
  tickers: z.string().min(1, "At least one ticker required"),
  model_type: z.string().min(1),
  horizon_days: z.coerce.number().int().min(1).max(60),
  train_ratio: z.coerce.number().min(0.5).max(0.9).default(0.7),
  val_ratio: z.coerce.number().min(0.05).max(0.3).default(0.15),
  search_mode: z.enum(["none", "grid", "optuna", "optuna_multi"]).default("none"),
  search_trials: z.coerce.number().int().min(1).max(200).default(20),
});

export default function TrainForm() {
  const train = useTrain();
  const { data: modelTypes } = useModelTypes();
  const form = useForm({
    resolver: zodResolver(schema),
    defaultValues: {
      tickers: "AAPL",
      model_type: "logistic",
      horizon_days: 5,
      train_ratio: 0.7,
      val_ratio: 0.15,
      search_mode: "none",
      search_trials: 20,
    },
  });

  const onSubmit = (values) => {
    const payload = {
      tickers: values.tickers.split(",").map((t) => t.trim()).filter(Boolean),
      model_type: values.model_type,
      horizon_days: values.horizon_days,
      train_ratio: values.train_ratio,
      val_ratio: values.val_ratio,
      search_mode: values.search_mode,
      search_trials: values.search_trials,
      feature_groups: ["ta_basic", "momentum"],
    };
    train.mutate(payload);
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 max-w-2xl">
      <div>
        <Label htmlFor="tickers">Tickers (comma-separated)</Label>
        <Input id="tickers" placeholder="AAPL, MSFT" {...form.register("tickers")} />
        {form.formState.errors.tickers && (
          <p className="text-sm text-down mt-1">{form.formState.errors.tickers.message}</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Model type</Label>
          <Select value={form.watch("model_type")} onValueChange={(v) => form.setValue("model_type", v)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(() => {
                // Backend returns { types: [{ type, class_name, available }], total }
                // Also defensively handle plain string arrays
                const FALLBACK = ["logistic", "random_forest", "xgboost", "lightgbm", "catboost", "ensemble"];
                let list = FALLBACK;
                if (Array.isArray(modelTypes)) list = modelTypes;
                else if (Array.isArray(modelTypes?.types)) list = modelTypes.types.map((m) => m.type || m);
                return list.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>);
              })()}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="horizon">Horizon days</Label>
          <Input id="horizon" type="number" {...form.register("horizon_days")} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Search mode</Label>
          <Select value={form.watch("search_mode")} onValueChange={(v) => form.setValue("search_mode", v)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="grid">Grid</SelectItem>
              <SelectItem value="optuna">Optuna</SelectItem>
              <SelectItem value="optuna_multi">Optuna Multi-Objective</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="trials">Trials</Label>
          <Input id="trials" type="number" {...form.register("search_trials")} />
        </div>
      </div>

      <Button type="submit" disabled={train.isPending}>
        {train.isPending ? "Starting..." : "Start Training"}
      </Button>

      {train.error && <ErrorState error={train.error} />}
      {train.data && (
        <div className="text-sm text-up">
          Training started · run_id: <code className="font-mono">{train.data.run_id || "—"}</code>
        </div>
      )}
    </form>
  );
}
