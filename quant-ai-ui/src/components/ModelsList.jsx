export default function ModelsList({
  models = [],
  promotedId,
  onPromote,
  onRefresh,
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-medium">Registered Models</h3>
        <button
          onClick={onRefresh}
          className="border border-gray-600 text-gray-300 hover:text-white px-3 py-1 rounded text-sm"
        >
          Refresh
        </button>
      </div>

      {/* Promoted Badge */}
      {promotedId && (
        <div className="bg-yellow-900/30 border border-yellow-600 text-yellow-300 px-4 py-3 rounded mb-4 text-sm">
          🏆 <strong>Production Model:</strong> {promotedId.substring(0, 8)}...
        </div>
      )}

      {models.length === 0 ? (
        <p className="text-gray-500 italic text-sm">No models registered. Train one first!</p>
      ) : (
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-gray-700 text-left">
              <th className="text-gray-400 font-medium pb-2 pr-4">Status</th>
              <th className="text-gray-400 font-medium pb-2 pr-4">Name</th>
              <th className="text-gray-400 font-medium pb-2 pr-4">Type</th>
              <th className="text-gray-400 font-medium pb-2 pr-4">Tickers</th>
              <th className="text-gray-400 font-medium pb-2 pr-4">Metrics</th>
              <th className="text-gray-400 font-medium pb-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const isPromoted = model.id === promotedId;
              return (
                <tr
                  key={model.id}
                  className={`border-b border-gray-800 ${isPromoted ? "bg-yellow-900/10" : ""}`}
                >
                  {/* Status */}
                  <td className="py-2 pr-4">
                    {isPromoted ? (
                      <span className="bg-yellow-600 text-black px-2 py-0.5 rounded text-xs font-bold">
                        🏆 PROD
                      </span>
                    ) : (
                      <span className="bg-green-900/50 text-green-300 px-2 py-0.5 rounded text-xs">
                        Active
                      </span>
                    )}
                  </td>

                  {/* Name */}
                  <td className="py-2 pr-4">
                    <div className="text-white font-medium">{model.name}</div>
                    <div className="text-gray-500 text-xs font-mono">{model.id?.substring(0, 8)}...</div>
                  </td>

                  {/* Type */}
                  <td className="py-2 pr-4">
                    <span className="bg-blue-900/40 text-blue-300 px-2 py-0.5 rounded text-xs">
                      {model.model_type}
                    </span>
                  </td>

                  {/* Tickers */}
                  <td className="py-2 pr-4 text-gray-300">{model.tickers?.join(", ")}</td>

                  {/* Metrics */}
                  <td className="py-2 pr-4 text-xs">
                    {model.metrics && (
                      <div className="space-y-0.5">
                        {model.metrics.val_auc && (
                          <div className="text-gray-400">
                            AUC: <strong className="text-white">{model.metrics.val_auc}</strong>
                          </div>
                        )}
                        {model.metrics.val_f1 && (
                          <div className="text-gray-400">
                            F1: <strong className="text-white">{model.metrics.val_f1}</strong>
                          </div>
                        )}
                      </div>
                    )}
                  </td>

                  {/* Actions */}
                  <td className="py-2">
                    {!isPromoted && (
                      <button
                        onClick={() => onPromote(model.id)}
                        className="bg-green-700 hover:bg-green-600 text-white px-2 py-1 rounded text-xs"
                      >
                        Promote
                      </button>
                    )}
                    {isPromoted && (
                      <span className="text-gray-500 text-xs">In Production</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
