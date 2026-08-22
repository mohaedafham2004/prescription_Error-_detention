import React from "react";
import { AlertCircle, ShieldAlert } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full border-t border-slate-200 dark:border-[#1E2A44] bg-slate-50/80 dark:bg-[#0B1120] text-slate-500 dark:text-[#94A3B8] text-xs py-8 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-xl bg-white dark:bg-[#131B2E] border border-slate-200 dark:border-[#1E2A44] shadow-sm dark:shadow-none">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center flex-shrink-0">
              <ShieldAlert className="h-4 w-4 text-[#D97706] dark:text-[#FBBF24]" />
            </div>
            <p className="text-[12px] text-slate-700 dark:text-[#CBD5E1] leading-snug">
              <strong className="text-[#D97706] dark:text-[#FBBF24]">Clinical Safety Notice:</strong> All flagged drug interactions, dosage warnings, and therapeutic duplications are generated for professional pharmacist and physician review only.
            </p>
          </div>
          <span className="text-[11px] text-slate-400 dark:text-slate-500 font-mono flex-shrink-0">
            TrOCR + spaCy NER + Clinical Rules
          </span>
        </div>


        <div className="flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500 pt-2">
          <p>© 2026 Smart Prescription Error Detection NLP System. SLTC Research Project.</p>
          <div className="flex items-center gap-4 mt-2 sm:mt-0">
            <span>FastAPI Backend (:8000)</span>
            <span>•</span>
            <span>Next.js App (:3000)</span>
          </div>
        </div>

      </div>
    </footer>
  );
}
