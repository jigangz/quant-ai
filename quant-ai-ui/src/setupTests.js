import "@testing-library/jest-dom";

// Mock lightweight-charts
vi.mock("lightweight-charts", () => ({
  createChart: vi.fn(() => ({
    addCandlestickSeries: vi.fn(() => ({
      setData: vi.fn(),
      setMarkers: vi.fn(),
    })),
    applyOptions: vi.fn(),
    timeScale: vi.fn(() => ({ fitContent: vi.fn() })),
    remove: vi.fn(),
    resize: vi.fn(),
  })),
  CrosshairMode: { Normal: 1 },
  LineStyle: { Solid: 0 },
}));

// Mock WebSocket
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 1;
  }
  send() {}
  close() {}
  addEventListener() {}
  removeEventListener() {}
}
global.WebSocket = MockWebSocket;
