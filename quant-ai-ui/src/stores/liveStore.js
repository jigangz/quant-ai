import { create } from "zustand";

export const useLiveStore = create((set) => ({
  prices: {},
  connectionStatus: "disconnected",
  updatePrice: (ticker, price, ts) =>
    set((state) => ({
      prices: { ...state.prices, [ticker]: { price, ts } },
    })),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  clearPrices: () => set({ prices: {} }),
}));
