import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Explain from "./pages/Explain";
import Training from "./pages/Training";

function App() {
  const [page, setPage] = useState("training");

  console.log("API BASE =", import.meta.env.VITE_API_BASE);

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      {/* Navigation */}
      <nav style={styles.nav}>
        <span style={styles.logo}>⚡ Quant AI</span>
        <div style={styles.links}>
          <button
            onClick={() => setPage("dashboard")}
            style={page === "dashboard" ? styles.activeLink : styles.link}
          >
            📊 Dashboard
          </button>
          <button
            onClick={() => setPage("training")}
            style={page === "training" ? styles.activeLink : styles.link}
          >
            🎯 Training
          </button>
          <button
            onClick={() => setPage("explain")}
            style={page === "explain" ? styles.activeLink : styles.link}
          >
            🔍 Explain
          </button>
        </div>
      </nav>

      {/* Page Content */}
      <main style={styles.main}>
        {page === "dashboard" && <Dashboard />}
        {page === "training" && <Training />}
        {page === "explain" && <Explain />}
      </main>
    </div>
  );
}

const styles = {
  nav: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 24,
    paddingBottom: 16,
    borderBottom: "2px solid #eee",
  },
  logo: {
    fontSize: 20,
    fontWeight: "bold",
  },
  links: {
    display: "flex",
    gap: 8,
  },
  link: {
    background: "none",
    border: "none",
    padding: "8px 16px",
    cursor: "pointer",
    fontSize: 14,
    color: "#666",
  },
  activeLink: {
    background: "#007bff",
    color: "white",
    border: "none",
    padding: "8px 16px",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 14,
  },
  main: {
    minHeight: "calc(100vh - 100px)",
  },
};

export default App;
