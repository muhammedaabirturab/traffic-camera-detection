import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import ResultView from "../components/Result.jsx";

export default function AnalysisPage() {
  const { id } = useParams();
  const [r, setR] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setR(null);
    setErr("");
    api.analysis(id).then(setR).catch((e) => setErr(e.message));
  }, [id]);
  return (
    <>
      <div className="page-head">
        <div className="eyebrow"><Link to="/history">← Detection history</Link></div>
        <h1>Saved analysis</h1>
      </div>
      {err && <div className="error-box">⚠ {err}</div>}
      {!r && !err && <div className="skeleton" style={{ height: 320 }} />}
      {r && <ResultView result={r} />}
    </>
  );
}
