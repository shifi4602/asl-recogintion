"use client";

import { useState, FormEvent } from "react";
import { startTraining } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import type { TrainRequestBody, TrainResponse } from "@/types/api";

const BACKBONES = [
  { value: "mobilenet_v2", label: "MobileNetV2 (fast, ~14MB)" },
  { value: "efficientnet_b0", label: "EfficientNetB0 (accurate, ~20MB)" },
];

export default function TrainPage() {
  const token = useAuthStore((s) => s.token);

  const [form, setForm] = useState<TrainRequestBody>({
    phase1_epochs: 10,
    phase2_epochs: 5,
    batch_size: 32,
    model_name: "asl_model",
    backbone: "mobilenet_v2",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState("");

  function update<K extends keyof TrainRequestBody>(key: K, value: TrainRequestBody[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const res = await startTraining(token!, form);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start training");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Train Model</h1>
        <p className="text-slate-400 mt-1">Configure and launch a training run (admin only)</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Form */}
        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Backbone */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">Backbone</label>
              <select
                className="input"
                value={form.backbone}
                onChange={(e) => update("backbone", e.target.value)}
              >
                {BACKBONES.map((b) => (
                  <option key={b.value} value={b.value}>{b.label}</option>
                ))}
              </select>
            </div>

            {/* Model name */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">Model name</label>
              <input
                className="input"
                type="text"
                value={form.model_name}
                onChange={(e) => update("model_name", e.target.value)}
                placeholder="asl_model"
                required
              />
            </div>

            {/* Phase 1 epochs */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Phase 1 epochs (frozen backbone)
                <span className="ml-2 text-brand-400 font-bold">{form.phase1_epochs}</span>
              </label>
              <input
                type="range" min={1} max={30}
                value={form.phase1_epochs}
                onChange={(e) => update("phase1_epochs", Number(e.target.value))}
                className="w-full accent-brand-500"
              />
              <div className="flex justify-between text-xs text-slate-600 mt-1"><span>1</span><span>30</span></div>
            </div>

            {/* Phase 2 epochs */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Phase 2 epochs (fine-tune, 0 = skip)
                <span className="ml-2 text-brand-400 font-bold">{form.phase2_epochs}</span>
              </label>
              <input
                type="range" min={0} max={20}
                value={form.phase2_epochs}
                onChange={(e) => update("phase2_epochs", Number(e.target.value))}
                className="w-full accent-brand-500"
              />
              <div className="flex justify-between text-xs text-slate-600 mt-1"><span>0</span><span>20</span></div>
            </div>

            {/* Batch size */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">Batch size</label>
              <select className="input" value={form.batch_size} onChange={(e) => update("batch_size", Number(e.target.value))}>
                {[8, 16, 32, 64, 128].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-sm">
                {error}
              </div>
            )}

            <button type="submit" className="btn-primary w-full" disabled={loading}>
              {loading && <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
              {loading ? "Submitting…" : "Start Training"}
            </button>
          </form>
        </div>

        {/* Status / Info */}
        <div className="flex flex-col gap-4">
          {result && (
            <div className="card animate-fade-in border-brand-500/30">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse-slow" />
                <span className="text-brand-400 font-semibold text-sm uppercase tracking-wide">Job Accepted</span>
              </div>
              <p className="text-slate-200 text-sm">{result.message}</p>
              <p className="text-slate-500 text-xs mt-3">
                Training runs in the background. Monitor progress in the MLflow UI at{" "}
                <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="text-brand-400 hover:underline">
                  localhost:5000
                </a>
              </p>
            </div>
          )}

          <div className="card bg-slate-900/50">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Training Pipeline</p>
            <ol className="space-y-3 text-sm text-slate-300">
              {[
                ["Phase 1", "Backbone frozen, train head layers only"],
                ["Phase 2", "Unfreeze top-30 backbone layers, fine-tune at LR=1e-5"],
                ["Evaluate", "Confusion matrix + accuracy on held-out test set"],
                ["Save", "Model saved as .keras to /models/"],
              ].map(([title, desc], i) => (
                <li key={i} className="flex gap-3">
                  <span className="w-5 h-5 rounded-full bg-brand-500/20 border border-brand-500/30 text-brand-400 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                    {i + 1}
                  </span>
                  <div>
                    <p className="font-medium text-white">{title}</p>
                    <p className="text-slate-400 text-xs">{desc}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
