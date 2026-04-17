import { useState } from "react";
import PageHeader from "../components/PageHeader";
import { Card } from "../components/ui/card";
import StrategyPicker from "../features/strategy/StrategyPicker";
import StrategyParamsForm from "../features/strategy/StrategyParamsForm";
import BacktestResults from "../features/strategy/BacktestResults";

export default function StrategyPage() {
  const [name, setName] = useState("ma_crossover");
  const [result, setResult] = useState(null);

  return (
    <div>
      <PageHeader title="Strategy" subtitle="Rule-based strategies with schema-driven parameters" />
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="p-3 lg:col-span-1">
          <StrategyPicker selected={name} onSelect={setName} />
        </Card>
        <div className="lg:col-span-3 space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Parameters — {name}</h3>
            <StrategyParamsForm name={name} onResult={setResult} />
          </Card>
          <BacktestResults result={result} />
        </div>
      </div>
    </div>
  );
}
