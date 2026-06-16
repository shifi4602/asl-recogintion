"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/auth-store";
import { useEffect } from "react";

const NAV = [
  { href: "/dashboard",        label: "Predict",     icon: "🔍" },
  { href: "/dashboard/stream", label: "Live Stream",  icon: "📡" },
  { href: "/dashboard/train",  label: "Train Model",  icon: "🏋️" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { token, username, clearAuth } = useAuthStore();

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  if (!token) return null;

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col">
        {/* Brand */}
        <div className="px-5 py-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🤟</span>
            <div>
              <p className="font-bold text-white text-sm leading-tight">ASL Recognition</p>
              <p className="text-slate-500 text-xs">v0.1.0</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ href, label, icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                  active
                    ? "bg-brand-500/15 text-brand-400 border border-brand-500/25"
                    : "text-slate-400 hover:text-white hover:bg-slate-800"
                }`}
              >
                <span className="text-base">{icon}</span>
                {label}
              </Link>
            );
          })}
        </nav>

        {/* User */}
        <div className="px-3 py-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-3 py-2">
            <div className="w-8 h-8 rounded-full bg-brand-500/20 border border-brand-500/30 flex items-center justify-center text-brand-400 font-bold text-sm">
              {username?.[0]?.toUpperCase() ?? "U"}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{username}</p>
              <p className="text-xs text-slate-500">admin</p>
            </div>
            <button
              onClick={() => { clearAuth(); router.push("/login"); }}
              title="Sign out"
              className="text-slate-500 hover:text-red-400 transition-colors text-sm"
            >
              ⏻
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-slate-950">
        <div className="max-w-4xl mx-auto p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
