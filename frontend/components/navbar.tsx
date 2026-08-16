"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Cpu, FileText, Info, ShieldCheck, Stethoscope, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function Navbar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = React.useState<"checking" | "online" | "offline">("checking");
  const [activeModels, setActiveModels] = React.useState<{ ocr: string; ner: string } | null>(null);

  React.useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    
    async function checkHealth() {
      try {
        const res = await fetch(`${apiUrl}/api/health`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          setBackendStatus("online");
          setActiveModels({ ocr: data.active_ocr_model, ner: data.active_ner_model });
        } else {
          setBackendStatus("offline");
        }
      } catch {
        setBackendStatus("offline");
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const navLinks = [
    { href: "/", label: "Prescription Analysis", icon: FileText },
    { href: "/performance", label: "Model Performance", icon: Activity },
    { href: "/about", label: "About & Architecture", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800 bg-slate-950/85 backdrop-blur-xl transition-all">
      <div className="max-w-7xl mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-sky-600 via-sky-500 to-teal-400 flex items-center justify-center shadow-lg shadow-sky-500/20 group-hover:scale-105 transition-transform">
              <Stethoscope className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base tracking-tight text-white group-hover:text-sky-300 transition-colors">
                  RxAssist
                </span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-sky-500/30 text-sky-400">
                  NLP v1.0
                </Badge>
              </div>
              <p className="text-[11px] text-slate-400 leading-none hidden sm:block">
                Smart Prescription Error Detection
              </p>
            </div>
          </Link>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-900/60 p-1.5 rounded-xl border border-slate-800/80">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? "bg-slate-800 text-sky-300 shadow-sm border border-slate-700/70"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-sky-400" : "text-slate-400"}`} />
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Status Indicators & Docs */}
        <div className="flex items-center gap-3">
          {/* Live Backend Connection Status */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/80 border border-slate-800 text-xs">
            <span
              className={`h-2 w-2 rounded-full animate-pulse ${
                backendStatus === "online"
                  ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"
                  : backendStatus === "checking"
                  ? "bg-amber-400"
                  : "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]"
              }`}
            />
            <span className="text-[11px] font-medium text-slate-300">
              API {backendStatus === "online" ? "Online" : backendStatus === "checking" ? "Connecting..." : "Offline"}
            </span>
            {activeModels && (
              <span className="text-[10px] text-slate-500 font-mono border-l border-slate-800 pl-2">
                {activeModels.ocr.toUpperCase()} • {activeModels.ner.toUpperCase()}
              </span>
            )}
          </div>

          {/* FastAPI Swagger Link */}
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:border-slate-700 text-xs font-medium transition-all"
          >
            <span>API Docs</span>
            <ExternalLink className="h-3 w-3 text-slate-500" />
          </a>
        </div>
      </div>

      {/* Mobile nav row */}
      <div className="flex md:hidden border-t border-slate-800/80 bg-slate-950 px-4 py-2 justify-around">
        {navLinks.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex flex-col items-center gap-1 py-1 px-3 text-[11px] font-medium ${
                isActive ? "text-sky-400" : "text-slate-400"
              }`}
            >
              <Icon className="h-4 w-4" />
              {link.label}
            </Link>
          );
        })}
      </div>
    </header>
  );
}
