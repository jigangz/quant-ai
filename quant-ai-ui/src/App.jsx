import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Explain from "./pages/Explain";

function App() {
  
  const [page, setPage] = useState("dashboard");

  
  console.log("API BASE =", import.meta.env.VITE_API_BASE);

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      {/* ===== navigation ===== */}
      <nav style={{ marginBottom: 24 }}>
        <button
          onClick={() => setPage("dashboard")}
          style={{ marginRight: 12 }}
        >
          Dashboard
        </button>

        <button onClick={() => setPage("explain")}>
          Explain
        </button>
      </nav>

      {/* ===== switch ===== */}
      {page === "dashboard" && <Dashboard />}
      {page === "explain" && <Explain />}
    </div>
  );
}

export default App;
