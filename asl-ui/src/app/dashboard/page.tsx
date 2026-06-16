"use client";

import { useState, useCallback, useRef } from "react";
import { predictSign } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import type { PredictionResponse } from "@/types/api";

export default function PredictPage() {
  const token = useAuthStore((s) => s.token);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError("");
    setPreview(URL.createObjectURL(f));
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f?.type.startsWith("image/")) handleFile(f);
    },
    [handleFile]
  );

  async function handlePredict() {
    if (!file || !token) return;
    setLoading(true);
    setError("");
    try {
      const res = await predictSign(token, file);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Sign Prediction</h1>
        <p className="text-slate-400 mt-1">Upload a hand image to classify its ASL letter</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload zone */}
        <div className="card flex flex-col gap-4">
          <h2 className="font-semibold text-slate-200">Upload Image</h2>

          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onClick={() => inputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-xl cursor-pointer transition-colors min-h-48 flex flex-col items-center justify-center gap-3
              ${dragging ? "border-brand-500 bg-brand-500/5" : "border-slate-600 hover:border-brand-500/60 hover:bg-slate-700/30"}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="preview" className="max-h-40 rounded-lg object-contain" />
            ) : (
              <>
                <span className="text-4xl">🖼️</span>
                <p className="text-slate-400 text-sm text-center">
                  Drag & drop or <span className="text-brand-400">click to browse</span>
                </p>
                <p className="text-slate-600 text-xs">JPEG, PNG, WebP</p>
              </>
            )}
          </div>

          {file && (
            <p className="text-xs text-slate-500 truncate">
              {file.name} — {(file.size / 1024).toFixed(1)} KB
            </p>
          )}

          <button
            className="btn-primary"
            onClick={handlePredict}
            disabled={!file || loading}
          >
            {loading && <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
            {loading ? "Classifying…" : "Detect Sign"}
          </button>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Result */}
        <div className="card flex flex-col gap-4">
          <h2 className="font-semibold text-slate-200">Result</h2>

          {result ? (
            <div className="animate-fade-in flex flex-col gap-5">
              {/* Big letter */}
              <div className="flex flex-col items-center py-6 bg-slate-900 rounded-xl border border-slate-700">
                <span className="text-8xl font-black text-brand-400 leading-none">
                  {result.sign}
                </span>
                <span className="text-slate-400 text-sm mt-2">{result.confidence_pct} confidence</span>
              </div>

              {/* Confidence bar */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Confidence</span>
                  <span>{result.confidence_pct}</span>
                </div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full transition-all duration-700"
                    style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>

              {/* Top-K */}
              <div>
                <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Top predictions</p>
                <div className="space-y-2">
                  {result.top_k.map((item, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="w-6 text-center font-bold text-slate-300">{item.sign}</span>
                      <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-brand-500/60 rounded-full"
                          style={{ width: `${(item.confidence * 100).toFixed(1)}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-400 w-12 text-right">
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <p className="text-xs text-slate-600 text-right">⚡ {result.latency_ms} ms</p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-12 text-slate-600">
              <span className="text-5xl mb-3">✋</span>
              <p className="text-sm">Upload an image to see the prediction</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
