import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import * as Dialog from "@radix-ui/react-dialog";
import { HelpCircle, X } from "lucide-react";
import { ragAnswer } from "@/api/client";

export function GlobalRagButton({ bottom = 24, right = 24 }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const ask = useMutation({ mutationFn: (q) => ragAnswer({ query: q, top_k: 5 }) });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    ask.mutate(query.trim());
  };

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <button
          aria-label="RAG Q&A"
          style={{ bottom, right }}
          className="fixed w-14 h-14 rounded-full bg-accent text-accent-foreground shadow-lg hover:scale-105 transition-transform z-50 flex items-center justify-center"
        >
          <HelpCircle size={24} />
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40 z-40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] max-w-[90vw] bg-surface border border-surface-border rounded-lg p-6 z-50 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <Dialog.Title className="text-lg font-bold text-foreground">问我任何量化问题</Dialog.Title>
            <Dialog.Close aria-label="Close" className="text-muted hover:text-foreground">
              <X size={20} />
            </Dialog.Close>
          </div>
          <form onSubmit={handleSubmit} className="flex gap-2 mb-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="问问题... 例如 RSI 超买是什么意思"
              className="flex-1 px-3 py-2 bg-surface-muted border border-surface-border rounded text-sm text-foreground placeholder:text-muted focus:outline-none focus:ring-1 focus:ring-accent"
            />
            <button
              type="submit"
              className="bg-accent text-accent-foreground px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
              disabled={ask.isPending}
            >
              {ask.isPending ? "查询中..." : "问"}
            </button>
          </form>
          {ask.data && (
            <div className="text-sm text-foreground">
              <div className="mb-2 p-3 bg-surface-muted rounded">{ask.data.answer}</div>
              {ask.data.evidence?.length > 0 && (
                <details className="text-xs text-muted">
                  <summary className="cursor-pointer">引用来源 ({ask.data.evidence.length})</summary>
                  <ul className="mt-2 space-y-1">
                    {ask.data.evidence.map((e) => (
                      <li key={e.id}>· {e.type}: {e.text?.slice(0, 120)}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}
          {ask.isError && (
            <div className="text-sm text-down">问答服务暂不可用，请稍后重试。</div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
