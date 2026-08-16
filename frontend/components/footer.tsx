import React from "react";
import { AlertCircle, ShieldAlert } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full border-t border-slate-900 bg-slate-950/90 text-slate-400 text-xs py-8 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center flex-shrink-0">
              <ShieldAlert className="h-4 w-4 text-amber-400" />
            </div>
            <p className="text-[12px] text-slate-300 leading-snug">
              <strong className="text-amber-400">Clinical Safety Notice:</strong> All flagged drug interactions, dosage warnings, and therapeutic duplications are generated for professional pharmacist and physician review only.
            </p>
          </div>
          <span className="text-[11px] text-slate-500 font-mono flex-shrink-0">
            TrOCR + spaCy NER + Clinical Rules
          </span>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500 pt-2">
          <p>© 2026 Smart Prescription Error Detection NLP System. SLTC Research Project.</p>
          <div className="flex items-center gap-4 mt-2 sm:mt-0">
            <span>FastAPI Backend (:8000)</span>
            <span>•</span>
            <span>Next.js App (:3000)</span>
            <span>•</span>
            <span>Streamlit Dashboard (:8501)</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
