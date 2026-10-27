import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const today = () => new Date().toISOString().slice(0, 10);

export default function App() {
  const [cfg, setCfg] = useState(null);
  const [ticker, setTicker] = useState("NVDA");
  const [date, setDate] = useState(today());
  const [analysts, setAnalysts] = useState([]);
  const [run, setRun] = useState(null);
  const [error, setError] = useState("");
  const timer = useRef(null);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((c) => {
        setCfg(c);
        setAnalysts(c.analysts);
      })
      .catch(() => setError("Cannot reach the API. Is uvicorn api:app running?"));
    return () => clearInterval(timer.current);
  }, []);

  const toggle = (a) =>
    setAnalysts((cur) => (cur.includes(a) ? cur.filter((x) => x !== a) : [...cur, a]));

  const poll = (id) => {
    clearInterval(timer.current);
    timer.current = setInterval(async () => {
      try {
        const job = await (await fetch(`/api/runs/${id}`)).json();
        setRun(job);
        if (job.status !== "running") clearInterval(timer.current);
      } catch {
        clearInterval(timer.current);
        setError("Lost connection to the API.");
      }
    }, 3000);
  };

  const start = async () => {
    setError("");
    setRun({ status: "running" });
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, date, analysts }),
    });
    if (!res.ok) {
      setRun(null);
      setError((await res.json()).detail || "Failed to start run");
      return;
    }
    poll((await res.json()).id);
  };

  const running = run?.status === "running";

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>Run settings</h2>
        <label>
          Ticker
          <input value={ticker} onChange={(e) => setTicker(e.target.value)} />
        </label>
        <label>
          Analysis date
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <fieldset>
          <legend>Analysts</legend>
          {cfg?.analysts.map((a) => (
            <label key={a} className="check">
              <input type="checkbox" checked={analysts.includes(a)} onChange={() => toggle(a)} />
              {a}
            </label>
          ))}
        </fieldset>
        {cfg && (
          <p className="muted">
            {cfg.provider} · deep: {cfg.deep_model} · quick: {cfg.quick_model}
          </p>
        )}
        <button onClick={start} disabled={running || !ticker.trim() || !analysts.length}>
          {running ? "Running…" : "Run analysis"}
        </button>
      </aside>

      <main>
        <h1>TradingAgents</h1>
        {error && <div className="box err">{error}</div>}
        {!run && !error && <p className="muted">Pick a ticker and date, then run the analysis.</p>}
        {running && (
          <div className="box">
            <span className="spinner" /> Running agents for {ticker} on {date}… this can take
            several minutes.
          </div>
        )}
        {run?.status === "error" && <div className="box err">{run.error}</div>}
        {run?.status === "done" && (
          <>
            <section className="box">
              <h2>Final decision</h2>
              <ReactMarkdown>{run.decision}</ReactMarkdown>
            </section>
            {Object.entries(run.reports).map(([k, v]) => (
              <details key={k} className="box">
                <summary>{k.replace(/_/g, " ")}</summary>
                <ReactMarkdown>{v}</ReactMarkdown>
              </details>
            ))}
          </>
        )}
      </main>
    </div>
  );
}
