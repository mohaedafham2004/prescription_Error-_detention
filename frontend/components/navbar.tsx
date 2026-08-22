"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Cpu, FileText, Info, ShieldCheck, Stethoscope, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/theme-toggle";


export function Navbar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = React.useState<"checking" | "online" | "offline">("checking");
  const [activeModels, setActiveModels] = React.useState<{ ocr: string; ner: string } | null>(null);

  React.useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`/api/health`, { method: "GET" });
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
    <header className="sticky top-0 z-50 w-full border-b border-slate-200 dark:border-[#1E2A44] bg-white/90 dark:bg-[#0B1120]/90 backdrop-blur-xl transition-all">
      <div className="max-w-7xl mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-[#0284C7] to-[#38BDF8] flex items-center justify-center shadow-md shadow-sky-500/20 group-hover:scale-105 transition-transform">
              <Stethoscope className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base tracking-tight text-slate-900 dark:text-white group-hover:text-[#0284C7] dark:group-hover:text-[#38BDF8] transition-colors">
                  Safe Prescription
                </span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-sky-500/30 text-[#0284C7] dark:text-[#38BDF8]">
                  NLP v1.0
                </Badge>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-[#94A3B8] leading-none hidden sm:block">
                Smart Prescription Error Detection
              </p>
            </div>
          </Link>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-100 dark:bg-[#131B2E] p-1.5 rounded-xl border border-slate-200 dark:border-[#1E2A44]">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? "bg-white dark:bg-[#1E2A44] text-[#0284C7] dark:text-[#38BDF8] shadow-sm border border-slate-200 dark:border-[#1E2A44]"
                    : "text-slate-600 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/50 dark:hover:bg-[#1E2A44]/40"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-[#0284C7] dark:text-[#38BDF8]" : "text-slate-400 dark:text-slate-500"}`} />
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Right Actions: Theme Toggle & API Docs */}
        <div className="flex items-center gap-2.5">
          <ThemeToggle />

          {/* FastAPI Swagger Link */}
          <a
          href="https://prescription-error-detention.onrender.com/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white dark:bg-[#131B2E] border border-slate-200 dark:border-[#1E2A44] text-slate-700 dark:text-[#E2E8F0] hover:text-[#0284C7] dark:hover:text-[#38BDF8] hover:border-sky-500/40 text-xs font-medium transition-all shadow-sm"
          >
            <span>API Docs</span>
            <ExternalLink className="h-3 w-3 text-slate-400" />
          </a>
        </div>

      </div>

      {/* Mobile nav row */}
      <div className="flex md:hidden border-t border-slate-200 dark:border-[#1E2A44] bg-white dark:bg-[#0B1120] px-4 py-2 justify-around">
        {navLinks.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex flex-col items-center gap-1 py-1 px-3 text-[11px] font-medium ${
                isActive ? "text-[#0284C7] dark:text-[#38BDF8]" : "text-slate-500 dark:text-[#94A3B8]"
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
