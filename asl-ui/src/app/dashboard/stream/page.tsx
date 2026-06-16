"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createStreamWebSocket } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import type { StreamMessage } from "@/types/api";

type Status = "idle" | "connecting" | "connected" | "error";

const STATUS_COLORS: Record<Status, string> = {
  idle: "bg-slate-500",
  connecting: "bg-yellow-500 animate-pulse",
  connected: "bg-brand-500",
  error: "bg-red-500",
};

const STATUS_LABELS: Record<Status, string> = {
  idle: "Disconnected",
  connecting: "Connecting…",
  connected: "Live",
  error: "Error",
};

export default function StreamPage() {
  const token = useAuthStore((s) => s.token);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<StreamMessage | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const stopStream = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setStatus("idle");
  }, []);

  const startStream = useCallback(async () => {
    if (!token) return;
    try {
      setResult(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus("connecting");
      const ws = createStreamWebSocket(token);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
        setIsStreaming(true);
        intervalRef.current = setInterval(() => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const canvas = canvasRef.current;
          const video = videoRef.current;
          if (!canvas || !video) return;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (blob && ws.readyState === WebSocket.OPEN) {
                blob.arrayBuffer().then((buf) => ws.send(buf));
              }
            },
            "image/jpeg",
            0.7
          );
        }, 150);
      };

      ws.onmessage = (e) => {
        try {
          const msg: StreamMessage = JSON.parse(e.data as string);
          setResult(msg);
          if (msg.error) {
            setStatus("error");
            return;
          }
          setStatus("connected");
          if (msg.sign && !msg.error) {
            setHistory((h) => [msg.sign!, ...h].slice(0, 10));
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onerror = () => setStatus("error");
      ws.onclose = () => {
        setStatus("idle");
        setIsStreaming(false);
      };
    } catch {
      setStatus("error");
    }
  }, [token]);

  useEffect(() => () => stopStream(), [stopStream]);

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Live Stream</h1>
        <p className="text-slate-400 mt-1">Real-time ASL recognition via your webcam</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Webcam feed */}
        <div className="lg:col-span-2 card flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-slate-200">Camera Feed</h2>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${STATUS_COLORS[status]}`} />
              <span className="text-xs text-slate-400">{STATUS_LABELS[status]}</span>
            </div>
          </div>

          <div className="relative bg-slate-900 rounded-xl overflow-hidden aspect-video">
            <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />
            <canvas ref={canvasRef} width={640} height={480} className="hidden" />

            {!isStreaming && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <span className="text-5xl">📷</span>
                  <p className="text-slate-500 text-sm mt-2">Camera not started</p>
                </div>
              </div>
            )}

            {/* Prediction overlay */}
            {result?.sign && isStreaming && (
              <div className="absolute top-3 left-3 bg-black/70 backdrop-blur rounded-xl px-4 py-2 border border-brand-500/30">
                <span className="text-4xl font-black text-brand-400">{result.sign}</span>
                {result.confidence_pct && (
                  <span className="text-xs text-slate-300 ml-2">{result.confidence_pct}</span>
                )}
              </div>
            )}

            {result?.error && (
              <div className="absolute bottom-3 left-3 right-3 bg-red-500/20 border border-red-500/30 rounded-lg px-3 py-2">
                <p className="text-red-400 text-xs">{result.error}</p>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            {!isStreaming ? (
              <button onClick={startStream} className="btn-primary flex-1">
                Start Camera
              </button>
            ) : (
              <button onClick={stopStream} className="btn-ghost flex-1 border border-slate-600">
                Stop Camera
              </button>
            )}
          </div>
        </div>

        {/* Side panel */}
        <div className="flex flex-col gap-4">
          {/* Current prediction */}
          <div className="card text-center">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Current Sign</p>
            <div className="text-7xl font-black text-brand-400 leading-none min-h-20 flex items-center justify-center">
              {result?.sign ?? "—"}
            </div>
            {result?.confidence !== undefined && (
              <div className="mt-3">
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full transition-all duration-300"
                    style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-1">{result.confidence_pct}</p>
              </div>
            )}
            {result?.latency_ms !== undefined && (
              <p className="text-xs text-slate-600 mt-2">Latency: {result.latency_ms} ms</p>
            )}
          </div>

          {/* History */}
          <div className="card flex-1">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Recent Signs</p>
            {history.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {history.map((s, i) => (
                  <span
                    key={i}
                    className="w-9 h-9 flex items-center justify-center rounded-lg text-lg font-bold border border-slate-700 text-slate-200"
                    style={{ opacity: 1 - i * 0.08 }}
                  >
                    {s}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-slate-600 text-sm">No signs detected yet</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
