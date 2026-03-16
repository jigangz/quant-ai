import { useState, useEffect } from 'react';
import { X, ArrowUpCircle, Eye, Cpu } from 'lucide-react';
import Card from '../components/UI/Card';
import Badge from '../components/UI/Badge';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import { fetchModels, promoteModel, type ModelInfo } from '../api';

const statusConfig: Record<string, { variant: 'success' | 'info' | 'neutral'; label: string }> = {
  production: { variant: 'success', label: 'Production' },
  candidate: { variant: 'info', label: 'Candidate' },
  archived: { variant: 'neutral', label: 'Archived' },
};

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [promoting, setPromoting] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    setLoading(true);
    try {
      const data = await fetchModels();
      setModels(data);
    } catch {
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  }

  async function handlePromote(id: string) {
    setPromoting(id);
    try {
      await promoteModel(id);
      await loadModels();
    } catch {
      // silent
    } finally {
      setPromoting(null);
    }
  }

  if (loading) return <LoadingSpinner text="Loading models..." />;

  if (error) {
    return (
      <div className="flex items-center justify-center py-24 text-gray-600">
        <div className="text-center">
          <div className="text-4xl mb-3">⚠️</div>
          <div className="text-sm">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-4 animate-fade-in">
      {/* Main table */}
      <div className={`${selectedModel ? 'flex-1' : 'w-full'} transition-all`}>
        <Card
          title={`Models (${models.length})`}
          action={
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Cpu className="w-3 h-3" />
              Model Registry
            </div>
          }
        >
          {models.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-gray-600">
              <div className="text-4xl mb-3">🤖</div>
              <div className="text-sm">No models registered</div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-dark-border text-xs text-gray-500">
                    <th className="text-left px-4 py-3 font-medium">Name</th>
                    <th className="text-left px-4 py-3 font-medium">Version</th>
                    <th className="text-right px-4 py-3 font-medium">AUC</th>
                    <th className="text-right px-4 py-3 font-medium">F1</th>
                    <th className="text-left px-4 py-3 font-medium">Trained</th>
                    <th className="text-center px-4 py-3 font-medium">Status</th>
                    <th className="text-right px-4 py-3 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => {
                    const status = statusConfig[m.status] || statusConfig.archived;
                    return (
                      <tr
                        key={m.id}
                        className={`border-b border-dark-border/50 hover:bg-dark-hover transition-colors cursor-pointer ${
                          selectedModel?.id === m.id ? 'bg-dark-hover' : ''
                        }`}
                        onClick={() => setSelectedModel(m)}
                      >
                        <td className="px-4 py-3 font-medium text-gray-200">{m.name}</td>
                        <td className="px-4 py-3 font-mono text-gray-400">{m.version}</td>
                        <td className="px-4 py-3 text-right font-mono text-gray-300">
                          {m.auc.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-gray-300">
                          {m.f1.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 text-gray-400 text-xs">
                          {new Date(m.trained_at).toLocaleDateString()}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <Badge variant={status.variant}>{status.label}</Badge>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <div className="flex items-center justify-end gap-2">
                            {m.status !== 'production' && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handlePromote(m.id);
                                }}
                                disabled={promoting === m.id}
                                className="flex items-center gap-1 px-2 py-1 text-xs bg-bull/20 text-bull hover:bg-bull/30 rounded-sm transition-colors disabled:opacity-50"
                              >
                                <ArrowUpCircle className="w-3 h-3" />
                                {promoting === m.id ? '...' : 'Promote'}
                              </button>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedModel(m);
                              }}
                              className="flex items-center gap-1 px-2 py-1 text-xs bg-accent/20 text-accent hover:bg-accent/30 rounded-sm transition-colors"
                            >
                              <Eye className="w-3 h-3" />
                              Details
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* Detail drawer */}
      {selectedModel && (
        <div className="w-80 flex-shrink-0 animate-slide-in">
          <Card className="sticky top-4">
            <div className="flex items-center justify-between px-4 py-3 border-b border-dark-border">
              <h3 className="text-sm font-semibold text-gray-200">Model Details</h3>
              <button
                onClick={() => setSelectedModel(null)}
                className="text-gray-500 hover:text-gray-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-4 flex flex-col gap-4">
              <div>
                <div className="text-xs text-gray-500 mb-0.5">Name</div>
                <div className="text-sm font-medium text-gray-200">{selectedModel.name}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-0.5">Version</div>
                <div className="text-sm font-mono text-gray-300">{selectedModel.version}</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">AUC</div>
                  <div className="text-lg font-mono font-bold text-accent">
                    {selectedModel.auc.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">F1 Score</div>
                  <div className="text-lg font-mono font-bold text-accent">
                    {selectedModel.f1.toFixed(3)}
                  </div>
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-0.5">Status</div>
                <Badge variant={statusConfig[selectedModel.status]?.variant || 'neutral'}>
                  {statusConfig[selectedModel.status]?.label || selectedModel.status}
                </Badge>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-0.5">Trained At</div>
                <div className="text-sm text-gray-400">
                  {new Date(selectedModel.trained_at).toLocaleString()}
                </div>
              </div>
              {selectedModel.description && (
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">Description</div>
                  <div className="text-sm text-gray-400 leading-relaxed">
                    {selectedModel.description}
                  </div>
                </div>
              )}
              {selectedModel.params && Object.keys(selectedModel.params).length > 0 && (
                <div>
                  <div className="text-xs text-gray-500 mb-1">Parameters</div>
                  <div className="bg-dark-bg rounded-sm p-2 text-xs font-mono text-gray-400 overflow-auto max-h-40">
                    {JSON.stringify(selectedModel.params, null, 2)}
                  </div>
                </div>
              )}
              {selectedModel.status !== 'production' && (
                <button
                  onClick={() => handlePromote(selectedModel.id)}
                  disabled={promoting === selectedModel.id}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-bull/20 text-bull hover:bg-bull/30 rounded-sm transition-colors disabled:opacity-50 text-sm font-medium"
                >
                  <ArrowUpCircle className="w-4 h-4" />
                  Promote to Production
                </button>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
