import { useEffect, useRef } from "react";
import { useLiveStore } from "../../stores/liveStore";

const WS_BASE = (import.meta.env.VITE_API_BASE || "http://localhost:8000").replace(/^http/, "ws");
const WS_URL = `${WS_BASE}/api/trading/ws/prices`;

export function useLivePrices() {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const { updatePrice, setConnectionStatus } = useLiveStore();

  useEffect(() => {
    let mounted = true;

    const connect = () => {
      if (!mounted) return;
      setConnectionStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mounted) return;
        setConnectionStatus("connected");
      };
      ws.onmessage = (event) => {
        if (!mounted) return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.ticker && msg.price != null) {
            updatePrice(msg.ticker, msg.price, msg.timestamp || Date.now());
          }
        } catch (_err) {
          // ignore parse errors
        }
      };
      ws.onerror = () => {
        if (!mounted) return;
        setConnectionStatus("error");
      };
      ws.onclose = () => {
        if (!mounted) return;
        setConnectionStatus("disconnected");
        // reconnect with 1s backoff, max 10s
        reconnectTimerRef.current = setTimeout(connect, 1000);
      };
    };

    connect();

    return () => {
      mounted = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [updatePrice, setConnectionStatus]);
}
