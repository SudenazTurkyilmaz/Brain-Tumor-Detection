import React, { useMemo, useState } from "react";
import "./App.css";
import { API_BASE } from "./api";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loadingPredict, setLoadingPredict] = useState(false);

  const [result, setResult] = useState(null);

  const canSend = useMemo(() => !!file, [file]);

  function onPickFile(e) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setResult(null);

    if (f) {
      const url = URL.createObjectURL(f);
      setPreviewUrl(url);
    } else {
      setPreviewUrl("");
    }
  }

  async function callPredict() {
    if (!file) return;
    setLoadingPredict(true);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Predict error: ${res.status}`);
      }

      const json = await res.json();
      setResult(json);
    } catch (err) {
      alert(`Predict hata: ${err.message}`);
    } finally {
      setLoadingPredict(false);
    }
  }

  return (
    <div className="page">
      <div className="container">
        <div className="header">
          <h1 className="title">Beyin Tümörü Tespiti</h1>
          <p className="subtitle">
            Görüntü yükle → sınıf tahmini al.
          </p>
        </div>

        <div className="grid">
          <section className="card">
            <h3 className="card-title">Görüntü Yükle</h3>

            <input
              className="file-input"
              type="file"
              accept="image/*"
              onChange={onPickFile}
            />

            {previewUrl && (
              <div className="preview">
                <div className="muted">Önizleme</div>
                <img src={previewUrl} alt="preview" className="image" />
              </div>
            )}

            <div className="actions">
              <button
                className="button primary"
                onClick={callPredict}
                disabled={!canSend || loadingPredict}
              >
                {loadingPredict ? "Tahmin..." : "Tahmin"}
              </button>
            </div>

            <div className="meta">API: {API_BASE}</div>
          </section>

          <section className="card">
            <h3 className="card-title">Sonuçlar</h3>

            {!result && (
              <div className="placeholder">
                Tahmin’e basınca sınıf ve olasılıklar burada görünecek.
              </div>
            )}

            {result && (
              <div>
                <div className="result-box">
                  <div className="muted">Tahmin</div>
                  <div className="result-class">{result.predicted_class}</div>
                </div>

                <h4 className="section-title">Olasılıklar</h4>
                <div className="prob-list">
                  {Object.entries(result.probabilities || {}).map(([k, v]) => {
                    const percentage = v * 100;
                    const barColor =
                      v > 0.6 ? "#10b981" : v > 0.3 ? "#f59e0b" : "#6366f1";

                    return (
                      <div key={k} className="prob-row">
                        <div className="prob-label">{k}</div>
                        <div className="prob-bar">
                          <div
                            className="prob-fill"
                            style={{
                              width: `${Math.round(percentage)}%`,
                              background: barColor,
                            }}
                          />
                        </div>
                        <div className="prob-value">{percentage.toFixed(2)}%</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

          </section>
        </div>

        <div className="footer-note">
          Not: Backend localhost:8000 çalışmıyorsa istekler başarısız olur.
        </div>
      </div>
    </div>
  );
}