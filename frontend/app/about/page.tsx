import * as React from "react";
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  FileCode,
  HardDrive,
  Info,
  Layers,
  Network,
  Pill,
  Server,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Terminal,
  Workflow,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function AboutPage() {
  const pipelineSteps = [
    {
      step: "01",
      title: "Preprocessing & Denoising",
      desc: "Deskew, adaptive threshold binarization, noise removal, and line/word bounding-box segmentation.",
      tech: "OpenCV • PIL • NumPy",
      color: "from-blue-500/20 to-sky-500/20 border-sky-500/30",
    },
    {
      step: "02",
      title: "Dual-Path Handwriting OCR",
      desc: "Primary Vision Transformer (TrOCR) line transcription with fallback custom character CNN.",
      tech: "HuggingFace TrOCR • PyTorch",
      color: "from-sky-500/20 to-teal-500/20 border-teal-500/30",
    },
    {
      step: "03",
      title: "Clinical Entity Extraction (NER)",
      desc: "Extracts structured entity spans: MEDICINE, DOSAGE, FREQUENCY, and DURATION from transcription.",
      tech: "spaCy Transition NLP • Posos/ClinicalNER",
      color: "from-teal-500/20 to-emerald-500/20 border-emerald-500/30",
    },
    {
      step: "04",
      title: "Clinical Safety & Error Checker",
      desc: "Audits dosage boundaries, drug-drug interaction pairs, therapeutic duplications, and spelling typos.",
      tech: "RapidFuzz • Pharmacopeia Rules",
      color: "from-emerald-500/20 to-amber-500/20 border-amber-500/30",
    },
    {
      step: "05",
      title: "Clinical Web Presentation",
      desc: "Real-time Next.js frontend and high-performance FastAPI backend REST endpoints.",
      tech: "FastAPI • Next.js 16 • Tailwind CSS",
      color: "from-amber-500/20 to-purple-500/20 border-purple-500/30",
    },

  ];

  const techStack = [
    { category: "Frontend", items: ["Next.js (App Router)", "TypeScript", "Tailwind CSS", "shadcn/ui", "Recharts", "Lucide React"] },
    { category: "Backend & API", items: ["FastAPI", "Uvicorn", "Pydantic v2", "Python Multipart", "CORS Middleware"] },
    { category: "Deep Learning & NLP", items: ["PyTorch", "Hugging Face Transformers", "Microsoft TrOCR", "spaCy 3.8", "RapidFuzz"] },
    { category: "Computer Vision", items: ["OpenCV (cv2)", "Pillow (PIL)", "NumPy", "Adaptive Thresholding"] },
    { category: "Evaluation & Benchmarks", items: ["jiwer (CER/WER)", "scikit-learn", "Precision/Recall/F1", "Confusion Matrices"] },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Hero Header */}
      <div className="border-b border-slate-800/80 pb-6">
        <div className="flex items-center gap-2 mb-2">
          <Badge variant="default" className="gap-1.5 py-1">
            <Info className="h-3.5 w-3.5 text-sky-400" />
            System Architecture & Documentation
          </Badge>
          <Badge variant="outline" className="text-slate-400">
            SLTC Research Project • 5th Sem NLP
          </Badge>
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          About Smart Prescription NLP
        </h1>
        <p className="mt-1 text-sm text-slate-400 max-w-3xl">
          An end-to-end intelligent clinical assistant engineered to prevent medication errors by reading handwritten prescriptions, structuring pharmacological entities, and performing real-time drug interaction safety checks.
        </p>
      </div>

      {/* Mission & Problem Statement */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="border-slate-800">
          <CardHeader>
            <CardTitle className="text-base text-slate-200 flex items-center gap-2">
              <Stethoscope className="h-5 w-5 text-sky-400" />
              Clinical Problem & Objective
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-300 leading-relaxed space-y-3">
            <p>
              Medication dispensing errors caused by illegible doctor handwriting, out-of-range dosing, unrecognized drug-drug interactions, and duplicate therapies represent a substantial source of preventable medical harm worldwide.
            </p>
            <p>
              This system solves this challenge by pairing computer vision handwriting models with structured natural language entity extraction and deterministic pharmacopeia safety rules.
            </p>
          </CardContent>
        </Card>

        <Card className="border-slate-800">
          <CardHeader>
            <CardTitle className="text-base text-slate-200 flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-teal-400" />
              Core Safety Capabilities
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-300 leading-relaxed space-y-2">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-teal-400 flex-shrink-0 mt-0.5" />
              <span><strong>Fuzzy Typo Resolution:</strong> Corrects handwriting OCR confusions (e.g. 0 ↔ o) and misspelling against reference databases.</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-teal-400 flex-shrink-0 mt-0.5" />
              <span><strong>Drug-Drug Interaction Detection:</strong> Flags high-risk combinations (Aspirin + Warfarin, Tramadol + SSRIs) with clinical recommendations.</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-teal-400 flex-shrink-0 mt-0.5" />
              <span><strong>Duplicate Prescribing Auditor:</strong> Identifies redundant prescriptions within 9 therapeutic classes (NSAIDs, PPIs, Statins, etc.).</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Visual Pipeline Flow */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Workflow className="h-5 w-5 text-sky-400" />
          <h2 className="text-xl font-bold text-white tracking-tight">
            End-to-End Processing Pipeline
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {pipelineSteps.map((s, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-xl border bg-gradient-to-b ${s.color} bg-slate-950 flex flex-col justify-between space-y-3 relative group hover:scale-[1.02] transition-transform`}
            >
              <div className="flex justify-between items-center">
                <span className="text-xs font-mono font-bold text-sky-400 bg-sky-950/80 px-2 py-0.5 rounded border border-sky-800/50">
                  STEP {s.step}
                </span>
                {idx < pipelineSteps.length - 1 && (
                  <ArrowRight className="h-4 w-4 text-slate-600 hidden md:block" />
                )}
              </div>
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-slate-100">{s.title}</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">{s.desc}</p>
              </div>
              <div className="pt-2 border-t border-slate-800/60 text-[10px] font-mono text-slate-400">
                {s.tech}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tech Stack Grid */}
      <Card className="border-slate-800">
        <CardHeader>
          <CardTitle className="text-base text-slate-200 flex items-center gap-2">
            <Layers className="h-5 w-5 text-teal-400" />
            Technology Stack & Frameworks
          </CardTitle>
          <CardDescription>
            Components powering the OCR engine, NLP pipelines, FastAPI REST backend, and Next.js frontend
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {techStack.map((cat, idx) => (
              <div key={idx} className="space-y-2.5 p-4 rounded-xl bg-slate-950 border border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-sky-400">
                  {cat.category}
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {cat.items.map((item, i) => (
                    <Badge key={i} variant="outline" className="text-[11px] text-slate-300 border-slate-700 bg-slate-900/60">
                      {item}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Hardware Constraints & Execution Notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="border-slate-800">
          <CardHeader>
            <CardTitle className="text-base text-slate-200 flex items-center gap-2">
              <Cpu className="h-5 w-5 text-amber-400" />
              Hardware Constraints & Deployment
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-300 space-y-2.5 leading-relaxed">
            <div className="flex justify-between border-b border-slate-800 pb-1.5">
              <span className="text-slate-400">Machine Specification:</span>
              <strong className="text-slate-200">ASUS ExpertBook (Intel Core i5, 24 GB RAM)</strong>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-1.5">
              <span className="text-slate-400">Local Inference:</span>
              <strong className="text-slate-200">CPU Execution (FastAPI + Next.js)</strong>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-1.5">
              <span className="text-slate-400">Model Fine-Tuning:</span>
              <strong className="text-slate-200">Google Colab (GPU T4/A100)</strong>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-800">
          <CardHeader>
            <CardTitle className="text-base text-slate-200 flex items-center gap-2">
              <Database className="h-5 w-5 text-purple-400" />
              Data Isolation Principles
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <p>
              Strict architectural separation is enforced across data folders:
            </p>
            <ul className="list-disc list-inside space-y-1 text-slate-400 text-[11px]">
              <li><code>data/characters/</code>: Single-character crops used strictly for CNN training.</li>
              <li><code>data/words_lines/</code>: Word/line crops used for TrOCR fine-tuning.</li>
              <li><code>data/ner/</code>: Span annotations used strictly for custom spaCy NER.</li>
              <li><code>data/error_rules/</code>: CSV reference tables for drugs, doses, and interactions.</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
