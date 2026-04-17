import { useState } from "react";
import PageHeader from "../components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Label } from "../components/ui/label";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import ShapFeatureList from "../features/explain/ShapFeatureList";
import SimilarCasesList from "../features/explain/SimilarCasesList";
import { useExplain, useSimilarCases } from "../api/queries";

export default function ExplainPage() {
  const [ticker, setTicker] = useState("AAPL");
  const [queryTicker, setQueryTicker] = useState("AAPL");

  const explain = useExplain(queryTicker);
  const search = useSimilarCases(queryTicker ? "high volatility rsi failed" : null);

  return (
    <div>
      <PageHeader title="Model Explainability" subtitle="SHAP feature importance + similar historical cases" />
      <form
        className="flex items-end gap-3 mb-6 max-w-md"
        onSubmit={(e) => {
          e.preventDefault();
          setQueryTicker(ticker.toUpperCase());
        }}
      >
        <div className="flex-1">
          <Label htmlFor="ticker">Ticker</Label>
          <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
        </div>
        <Button type="submit">Reload</Button>
      </form>

      {explain.isLoading && <LoadingOverlay label="Loading SHAP..." />}
      {explain.error && <ErrorState error={explain.error} onRetry={() => explain.refetch()} />}

      {!explain.isLoading && !explain.error && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader><CardTitle>SHAP Top Features</CardTitle></CardHeader>
            <CardContent>
              <ShapFeatureList
                features={
                  explain.data?.top_features ||
                  explain.data?.data?.top_features ||
                  []
                }
                message={explain.data?.data?.error || explain.data?.error}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Similar Historical Cases</CardTitle></CardHeader>
            <CardContent>
              {search.isLoading ? (
                <LoadingOverlay label="Searching..." />
              ) : (
                <SimilarCasesList
                  results={
                    Array.isArray(search.data)
                      ? search.data
                      : search.data?.results || []
                  }
                  message={search.data?.message}
                />
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
