"use client";

import * as React from "react";
import Image from "next/image";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Clock,
  Copy,
  Crop as CropIcon,
  Download,
  FileCheck2,
  FileSearch,
  FileText,
  HelpCircle,
  Info,
  Layers,
  Lightbulb,
  Pill,
  RefreshCw,
  RotateCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UploadCloud,
  X,
  ZoomIn,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { ImageCropperModal } from "@/components/image-cropper-modal";

import { Skeleton } from "@/components/ui/skeleton";

interface Entities {
  medicine: string[];
  dosage: string[];
  frequency: string[];
  duration: string[];
}

interface Issue {
  error_type: string;
  severity: "HIGH" | "MEDIUM" | "LOW" | "INFO";
  field: string;
  value: string;
  message: string;
  suggestion?: string;
}

interface LineItem {
  line_index: number;
  text: string;
  confidence: number;
  model_used?: string;
}

interface DrugMonograph {
  name: string;
  generic_name: string;
  brand_names?: string;
  composition?: string;
  manufacturer?: string;
  therapeutic_class?: string;
  dosage_forms?: string;
  usage?: string;
  standard_dosage?: string;
  precautions?: string;
}

interface AnalyzeResponse {
  extracted_text: string;
  entities: Entities;
  monographs?: Record<string, DrugMonograph>;
  issues: Issue[];
  risk?: {
    level: "clear" | "low" | "medium" | "high";
    reason: string;
    message: string;
  };
  lines: LineItem[];
  ocr_model_used: string;
  ocr_confidence: number;
  ner_confidence: number;
  ner_available: boolean;
  total_time_s: number;
  error?: string | null;
}

interface SampleItem {
  id: string;
  title: string;
  category: string;
  description: string;
  filename: string;
}

export default function PrescriptionAnalysisPage() {
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [activeSampleId, setActiveSampleId] = React.useState<string | null>(null);
  const [isDragging, setIsDragging] = React.useState<boolean>(false);

  const [samples, setSamples] = React.useState<SampleItem[]>([]);
  const [isLoading, setIsLoading] = React.useState<boolean>(false);
  const [loadingStep, setLoadingStep] = React.useState<string>("Preparing pipeline...");
  const [result, setResult] = React.useState<AnalyzeResponse | null>(null);
  const [selectedMonograph, setSelectedMonograph] = React.useState<DrugMonograph | null>(null);
  const [apiError, setApiError] = React.useState<string | null>(null);
  const [copiedText, setCopiedText] = React.useState<boolean>(false);

  const [isCropModalOpen, setIsCropModalOpen] = React.useState<boolean>(false);
  const [rawImageForCrop, setRawImageForCrop] = React.useState<string | null>(null);

  const fileInputRef = React.useRef<HTMLInputElement>(null);
  // Fetch sample catalog on mount
  React.useEffect(() => {
    async function loadSamples() {
      try {
        const res = await fetch(`/api/samples`);
        if (res.ok) {
          const data = await res.json();
          setSamples(data);
        }
      } catch {
        // Fallback local samples
        setSamples([
          {
            id: "cardio_rx",
            title: "Cardiology Prescription",
            category: "Cardiovascular",
            description: "Atorvastatin 40mg, Aspirin 75mg, Clopidogrel 75mg, Pantoprazole 40mg",
            filename: "cardio_rx.png",
          },
          {
            id: "infection_rx",
            title: "Infection & Respiratory Clinic",
            category: "Infectious Disease",
            description: "Augmentin 625mg, Paracetamol 650mg, Cetirizine 10mg, Salbutamol 100mcg",
            filename: "infection_rx.png",
          },
          {
            id: "diabetic_care_rx",
            title: "Metropolitan Diabetes Care",
            category: "Endocrinology",
            description: "Metformin 500mg, Lisinopril 10mg, Amlodipine 5mg, Multivitamin",
            filename: "diabetic_care_rx.png",
          },
          {
            id: "sample_rx",
            title: "General Antibiotic",
            category: "General Practice",
            description: "Amoxicillin 500mg TDS for 7 days",
            filename: "sample_rx.png",
          },
        ]);
      }
    }
    loadSamples();
  }, []);

  // Handle custom file selection
  const handleFileChange = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setApiError("Please select a valid image file (PNG, JPG, JPEG, WEBP).");
      return;
    }
    setApiError(null);
    setSelectedFile(file);
    setActiveSampleId(null);
    setResult(null);
    setSelectedMonograph(null);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setRawImageForCrop(url);
    setIsCropModalOpen(true);
  };

  const handleCropComplete = (croppedFile: File, croppedUrl: string) => {
    setSelectedFile(croppedFile);
    setPreviewUrl(croppedUrl);
    setIsCropModalOpen(false);
  };

  const handleCropCancel = () => {
    setIsCropModalOpen(false);
  };


  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  // Select a preset sample
  const handleSelectSample = async (sample: SampleItem) => {
    try {
      setApiError(null);
      setIsLoading(true);
      setLoadingStep(`Loading sample "${sample.title}"...`);
      setSelectedFile(null);
      setActiveSampleId(sample.id);
      setResult(null);
      setSelectedMonograph(null);

      const res = await fetch(`/api/sample-image/${sample.id}`);
      if (!res.ok) {
        throw new Error("Could not load sample prescription from backend.");
      }
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: "image/png" });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(blob));
    } catch (err: any) {
      setApiError(err.message || "Failed to load sample image.");
    } finally {
      setIsLoading(false);
    }
  };

  // Run pipeline analysis
  const handleAnalyze = async () => {
    if (!selectedFile && !activeSampleId) {
      setApiError("Please upload an image or select a sample prescription first.");
      return;
    }

    setApiError(null);
    setIsLoading(true);
    setResult(null);
    setSelectedMonograph(null);

    const steps = [
      "Applying CLAHE contrast equalization & deskewing...",
      "Executing TrOCR line transcription & typo calibration...",
      "Extracting medical entities (spaCy Clinical NER)...",
      "Auditing drug-drug interactions, dosage bounds & duplicate prescribing...",
      "Fetching clinical drug monographs & FDA monographs...",
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      stepIdx = (stepIdx + 1) % steps.length;
      setLoadingStep(steps[stepIdx]);
    }, 1100);

    try {
      const formData = new FormData();
      if (activeSampleId) {
        formData.append("sample_id", activeSampleId);
      } else if (selectedFile) {
        formData.append("file", selectedFile);
      }

      const res = await fetch(`/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errorData.detail || `Server error (${res.status})`);
      }

      const data: AnalyzeResponse = await res.json();
      if (data.error) {
        setApiError(data.error);
      } else {
        setResult(data);
        // Auto-select first monograph if available
        if (data.monographs && Object.keys(data.monographs).length > 0) {
          const firstKey = Object.keys(data.monographs)[0];
          setSelectedMonograph(data.monographs[firstKey]);
        }
      }
    } catch (err: any) {
      setApiError(err.message || "An unexpected error occurred during prescription analysis.");
    } finally {
      clearInterval(interval);
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    if (!result?.extracted_text) return;
    navigator.clipboard.writeText(result.extracted_text);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 2000);
  };

  const downloadPdf = async () => {
    if (!result) return;
    const { jsPDF } = await import("jspdf");
    const autoTable = (await import("jspdf-autotable")).default;

    const doc = new jsPDF();
    const now = new Date();
    const filename = selectedFile?.name || "prescription_scan.png";

    // Header
    doc.setFontSize(18);
    doc.setTextColor(15, 23, 42);
    doc.text("Prescription Analysis — Clinical Audit Report", 14, 18);

    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.text("AI-Assisted Prescription Error Detection & Safety Review", 14, 24);

    // Divider
    doc.setDrawColor(2, 132, 199);
    doc.setLineWidth(0.8);
    doc.line(14, 27, 196, 27);

    // Meta box
    const ocrConf = result.ocr_confidence > 0 ? `${(result.ocr_confidence * 100).toFixed(1)}%` : "N/A";
    const riskLevel = (result.risk?.level || "clear").toUpperCase();

    autoTable(doc, {
      startY: 31,
      head: [],
      body: [
        [
          `Report ID: RX-${now.toISOString().replace(/[-:T]/g, "").slice(0, 15)}`,
          `AI Vision Engine: ${result.ocr_model_used || "AI Vision Engine"}`
        ],
        [
          `Generated: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`,
          `OCR Confidence: ${ocrConf}`
        ],
        [
          `Source File: ${filename}`,
          `Risk Level: ${riskLevel}`
        ]
      ],
      theme: "plain",
      styles: { fontSize: 8.5, textColor: [51, 65, 85], cellPadding: 2 },
    });

    let currentY = (doc as any).lastAutoTable.finalY + 4;

    // Risk Banner
    const isHigh = riskLevel === "HIGH";
    const isMed = riskLevel === "MEDIUM";
    const isLow = riskLevel === "LOW";
    const bannerBg = isHigh ? [254, 226, 226] : isMed ? [254, 243, 199] : isLow ? [224, 242, 254] : [220, 252, 231];
    const bannerBorder = isHigh ? [239, 68, 68] : isMed ? [245, 158, 11] : isLow ? [2, 132, 199] : [34, 197, 94];
    const bannerText = isHigh ? [153, 27, 27] : isMed ? [146, 64, 14] : isLow ? [7, 89, 133] : [22, 101, 52];

    doc.setFillColor(bannerBg[0], bannerBg[1], bannerBg[2]);
    doc.setDrawColor(bannerBorder[0], bannerBorder[1], bannerBorder[2]);
    doc.roundedRect(14, currentY, 182, 16, 2, 2, "FD");

    doc.setFontSize(9);
    doc.setTextColor(bannerText[0], bannerText[1], bannerText[2]);
    doc.setFont("helvetica", "bold");
    doc.text(`CLINICAL RISK ASSESSMENT: ${riskLevel}`, 18, currentY + 6);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.text(result.risk?.message || "No critical contraindications detected.", 18, currentY + 11);

    currentY += 22;

    // 1. Prescribed Medications Table
    doc.setFontSize(11);
    doc.setTextColor(15, 23, 42);
    doc.setFont("helvetica", "bold");
    doc.text("1. Prescribed Medications Regimen", 14, currentY);

    const meds = result.entities?.medicine || [];
    const dosages = result.entities?.dosage || [];
    const freqs = result.entities?.frequency || [];
    const durs = result.entities?.duration || [];
    const maxItems = Math.max(meds.length, dosages.length, freqs.length, durs.length);

    const medTableBody = [];
    for (let i = 0; i < maxItems; i++) {
      medTableBody.push([
        String(i + 1),
        meds[i] || "Unidentified Drug",
        dosages[i] || "Not specified",
        freqs[i] || "Not specified",
        durs[i] || "Not specified",
      ]);
    }

    autoTable(doc, {
      startY: currentY + 3,
      head: [["#", "Medication Name", "Dosage", "Frequency", "Duration"]],
      body: medTableBody.length > 0 ? medTableBody : [["-", "No medicines detected", "-", "-", "-"]],
      headStyles: { fillColor: [15, 23, 42], textColor: [255, 255, 255], fontStyle: "bold", fontSize: 8.5 },
      styles: { fontSize: 8, cellPadding: 3 },
      alternateRowStyles: { fillColor: [248, 250, 252] },
    });

    currentY = (doc as any).lastAutoTable.finalY + 8;

    // 2. Safety Audit Table
    doc.setFontSize(11);
    doc.setTextColor(15, 23, 42);
    doc.setFont("helvetica", "bold");
    doc.text("2. Safety & Error Audit", 14, currentY);

    const issues = result.issues || [];
    const issueTableBody = issues.map((iss) => [
      (iss.severity || "INFO").toUpperCase(),
      iss.error_type || "NOTE",
      iss.value || iss.field || "-",
      `${iss.message || ""}${iss.suggestion ? `\nAction: ${iss.suggestion}` : ""}`,
    ]);

    autoTable(doc, {
      startY: currentY + 3,
      head: [["Severity", "Error Type", "Target", "Clinical Warning & Recommendation"]],
      body: issueTableBody.length > 0 ? issueTableBody : [["OK", "CLEAR", "All Fields", "No contraindications, dosage errors, or adverse interactions detected."]],
      headStyles: { fillColor: [51, 65, 85], textColor: [255, 255, 255], fontStyle: "bold", fontSize: 8.5 },
      styles: { fontSize: 8, cellPadding: 3 },
      alternateRowStyles: { fillColor: [248, 250, 252] },
    });



    // Disclaimer
    const pageHeight = doc.internal.pageSize.height;
    doc.setFontSize(6.5);
    doc.setTextColor(148, 163, 184);
    doc.text(
      "Clinical Disclaimer: This report is generated by an AI-assisted optical character recognition (OCR) and NLP error detection pipeline for clinical decision support.",
      14,
      pageHeight - 8
    );

    doc.save(`clinical_report_${Date.now()}.pdf`);
  };

  const downloadJson = () => {
    if (!result) return;

    const now = new Date();
    const meds = result.entities?.medicine || [];
    const dosages = result.entities?.dosage || [];
    const frequencies = result.entities?.frequency || [];
    const durations = result.entities?.duration || [];
    const monographs = result.monographs || {};

    const maxItems = Math.max(meds.length, dosages.length, frequencies.length, durations.length, 1);
    const prescribedMedications = [];

    for (let i = 0; i < maxItems; i++) {
      const medName = meds[i] || "";
      if (!medName && i >= meds.length && !dosages[i] && !frequencies[i] && !durations[i]) {
        continue;
      }
      const entry: any = {
        item_number: i + 1,
        medication_name: medName || "Unidentified Drug",
        dosage: dosages[i] || "Not specified",
        frequency: frequencies[i] || "Not specified",
        duration: durations[i] || "Not specified",
      };

      if (medName && monographs[medName]) {
        const mono = monographs[medName];
        entry.monograph_reference = {
          generic_name: mono.generic_name || medName,
          brand_names: mono.brand_names || "N/A",
          therapeutic_class: mono.therapeutic_class || "N/A",
          standard_dosage: mono.standard_dosage || "N/A",
          precautions: mono.precautions || "N/A",
        };
      }
      prescribedMedications.push(entry);
    }

    const formattedIssues = (result.issues || []).map((iss) => ({
      severity: (iss.severity || "INFO").toUpperCase(),
      error_type: iss.error_type || "CLINICAL_NOTE",
      affected_field: iss.field || "GENERAL",
      detected_value: iss.value || "",
      clinical_warning: iss.message || "",
      recommended_action: iss.suggestion || "Verify with prescribing physician.",
    }));

    const lineBreakdown = (result.lines || []).map((line, idx) => ({
      line_number: (line.line_index ?? idx) + 1,
      transcription: line.text || "",
      confidence: line.confidence > 0 ? `${(line.confidence * 100).toFixed(1)}%` : "N/A",
    }));

    const formattedReport = {
      report_metadata: {
        report_id: `RX-${now.toISOString().replace(/[-:T]/g, "").slice(0, 15)}`,
        generated_at_utc: now.toISOString(),
        source_document: selectedFile?.name || "prescription_scan.png",
        ai_vision_engine: result.ocr_model_used || "AI Vision Engine",
        overall_ocr_confidence: result.ocr_confidence > 0 ? `${(result.ocr_confidence * 100).toFixed(1)}%` : "N/A",
        processing_time_seconds: result.total_time_s || 0,
      },
      clinical_safety_assessment: {
        risk_level: (result.risk?.level || "clear").toUpperCase(),
        assessment_summary: result.risk?.reason || "No critical issues detected.",
        guidance_message: result.risk?.message || "Confirm with licensed pharmacist or doctor before use.",
        human_pharmacist_review_recommended: ["high", "medium"].includes((result.risk?.level || "").toLowerCase()),
      },
      prescribed_medications_summary: {
        total_medications_detected: prescribedMedications.length,
        medications: prescribedMedications,
      },
      safety_and_error_audit: {
        total_issues_flagged: formattedIssues.length,
        flagged_issues: formattedIssues,
      },
      transcription_details: {
        full_extracted_text: result.extracted_text || "",
        segmented_lines: lineBreakdown,
      },
    };

    const blob = new Blob([JSON.stringify(formattedReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `clinical_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Hero Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Badge variant="default" className="gap-1.5 py-1">
              <Sparkles className="h-3.5 w-3.5 text-sky-400" />
              High-Accuracy Clinical NLP + RxVision Engine
            </Badge>
            <Badge variant="outline" className="text-slate-400">
              TrOCR • CLAHE • spaCy NER • Drug Monographs
            </Badge>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Prescription Analysis
          </h1>
          <p className="mt-1 text-sm text-slate-400 max-w-3xl">
            Transcribe handwritten prescriptions, automatically extract clinical entities, audit for drug-drug interactions, and retrieve comprehensive pharmacological monographs.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {result && (
            <>
              <Button
                variant="default"
                size="sm"
                onClick={downloadPdf}
                className="gap-2 bg-sky-600 hover:bg-sky-500 text-white font-medium shadow-md shadow-sky-600/20"
              >
                <Download className="h-4 w-4" />
                Download PDF Report
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={downloadJson}
                className="gap-2 text-slate-300 border-slate-700 hover:bg-slate-800"
              >
                <Download className="h-4 w-4" />
                JSON
              </Button>
            </>
          )}
        </div>
      </div>



      {/* API Error Alert */}
      {apiError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Pipeline Error</AlertTitle>
          <AlertDescription>{apiError}</AlertDescription>
        </Alert>
      )}

      {/* Upload & Preview Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Upload Dropzone */}
        <div className={result ? "lg:col-span-4" : "lg:col-span-6"}>
          <Card className="h-full border-slate-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-base text-slate-200">
                <UploadCloud className="h-4 w-4 text-sky-400" />
                Upload Prescription Scan
              </CardTitle>
              <CardDescription>
                Upload any handwritten or printed doctor prescription image
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
              />

              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[180px] ${
                  isDragging
                    ? "border-sky-400 bg-sky-500/10 shadow-lg shadow-sky-500/10"
                    : "border-slate-800 hover:border-slate-700 bg-slate-950/50 hover:bg-slate-900/50"
                }`}
              >
                <div className="h-12 w-12 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                  <UploadCloud className="h-6 w-6 text-sky-400" />
                </div>
                <p className="text-sm font-semibold text-slate-200">
                  Browse or drag & drop prescription image
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Supports PNG, JPG, JPEG, WEBP scans
                </p>
              </div>

              {/* Selected File Details */}
              {selectedFile && (
                <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                  <div className="flex items-center gap-2.5 truncate">
                    <FileText className="h-4 w-4 text-sky-400 flex-shrink-0" />
                    <span className="font-medium text-slate-300 truncate">
                      {selectedFile.name}
                    </span>
                    {activeSampleId && (
                      <Badge variant="default" className="text-[10px] py-0">
                        Preset Sample
                      </Badge>
                    )}
                  </div>
                  <span className="text-slate-500 flex-shrink-0">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              )}

              {/* Analyze CTA */}
              <Button
                onClick={handleAnalyze}
                disabled={(!selectedFile && !activeSampleId) || isLoading}
                className="w-full h-11 text-sm font-bold gap-2 bg-gradient-to-r from-sky-600 to-teal-600 hover:from-sky-500 hover:to-teal-500 text-white shadow-lg shadow-sky-600/20"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Analyzing Prescription...
                  </>
                ) : (
                  <>
                    <FileSearch className="h-4 w-4" />
                    Analyze Prescription
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Prescription Image Preview */}
        <div className={result ? "lg:col-span-8" : "lg:col-span-6"}>
          <Card className="h-full border-slate-800">
            <CardHeader className="pb-3 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base text-slate-200">
                  <FileCheck2 className="h-4 w-4 text-teal-400" />
                  Prescription Document Preview
                </CardTitle>
                <CardDescription>Visual inspection of input scan</CardDescription>
              </div>
              {previewUrl && (
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setRawImageForCrop(previewUrl);
                      setIsCropModalOpen(true);
                    }}
                    className="text-xs h-7 gap-1.5 border-slate-700 text-sky-400 hover:bg-slate-800 hover:text-sky-300"
                  >
                    <CropIcon className="h-3 w-3" />
                    Crop / Adjust Scan
                  </Button>
                  <Badge variant="outline" className="text-slate-400 font-mono text-[10px]">
                    Document Ready
                  </Badge>
                </div>
              )}

            </CardHeader>
            <CardContent>
              {previewUrl ? (
                <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-950 max-h-[380px] flex items-center justify-center p-2">
                  <img
                    src={previewUrl}
                    alt="Prescription preview"
                    className="max-h-[350px] w-auto object-contain rounded-lg shadow-md"
                  />
                </div>
              ) : (
                <div className="h-[220px] rounded-xl border border-slate-800/80 bg-slate-950/40 flex flex-col items-center justify-center text-slate-500 gap-2 p-6 text-center">
                  <FileText className="h-8 w-8 text-slate-600" />
                  <p className="text-sm font-medium">No document selected</p>
                  <p className="text-xs text-slate-600 max-w-sm">
                    Select a sample preset above or upload an image scan to begin.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Loading Skeleton Indicator */}
      {isLoading && (
        <Card className="border-sky-500/30 bg-sky-950/20 shadow-xl animate-in fade-in-50">
          <CardContent className="py-8 space-y-4 text-center">
            <div className="inline-flex h-12 w-12 rounded-full bg-sky-500/10 border border-sky-500/30 items-center justify-center text-sky-400 animate-pulse">
              <RefreshCw className="h-6 w-6 animate-spin text-sky-400" />
            </div>
            <div>
              <h4 className="text-base font-bold text-slate-100">Running Enhanced Pipeline</h4>
              <p className="text-xs text-sky-300 font-mono mt-1">{loadingStep}</p>
            </div>
            <div className="max-w-md mx-auto space-y-2">
              <Progress value={75} className="h-2" />
              <div className="flex justify-between text-[11px] text-slate-500">
                <span>CLAHE & Deskew</span>
                <span>TrOCR & Typo Fixes</span>
                <span>NER & Safety Rules</span>
                <span>Drug Monographs</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pipeline Results Section */}
      {result && (
        <div className="space-y-8 animate-in fade-in duration-300">
          {/* Section Header */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-teal-400" />
              <h2 className="text-xl font-bold text-white tracking-tight">
                Analysis Results
              </h2>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <Clock className="h-3.5 w-3.5 text-slate-500" />
                Processing Time: <strong className="text-slate-200">{result.total_time_s}s</strong>
              </span>
            </div>

          </div>

          {/* Headline Overall Risk Assessment Banner */}
          {result.risk && (
            <Alert
              variant={
                result.risk.level === "high"
                  ? "destructive"
                  : result.risk.level === "medium"
                  ? "warning"
                  : result.risk.level === "low"
                  ? "info"
                  : "success"
              }
              className="p-5 shadow-xl border"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <AlertTitle className="text-base font-bold tracking-tight flex items-center gap-2">
                    <span>OVERALL RISK ASSESSMENT:</span>
                    <span className="uppercase text-sky-400">{result.risk.level}</span>
                  </AlertTitle>
                  <AlertDescription className="text-sm leading-relaxed text-slate-100 font-medium pt-1">
                    {result.risk.message}
                  </AlertDescription>
                  <p className="text-xs text-slate-300/90 italic pt-1">
                    Rationale: {result.risk.reason}
                  </p>
                </div>
                <Badge
                  variant={
                    result.risk.level === "high"
                      ? "destructive"
                      : result.risk.level === "medium"
                      ? "warning"
                      : result.risk.level === "low"
                      ? "default"
                      : "success"
                  }
                  className="text-xs font-bold uppercase tracking-wider px-3 py-1 flex-shrink-0"
                >
                  {result.risk.level} Risk
                </Badge>
              </div>
            </Alert>
          )}

          {/* Top Row: Extracted Text & Structured Entities */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Extracted Text Monospace Card */}
            <div className="lg:col-span-5">
              <Card className="h-full border-slate-800 flex flex-col justify-between">
                <CardHeader className="pb-3 flex flex-row items-center justify-between">
                  <div>
                    <CardTitle className="text-base text-slate-200">
                      <FileText className="h-4 w-4 text-sky-400" />
                      OCR Transcription
                    </CardTitle>
                    <CardDescription>
                      Calibrated text transcription
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyToClipboard}
                    className="h-8 gap-1.5 text-xs text-slate-400 hover:text-white"
                  >
                    <Copy className="h-3.5 w-3.5" />
                    {copiedText ? "Copied" : "Copy"}
                  </Button>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-between space-y-4">
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 font-mono text-xs text-slate-200 whitespace-pre-wrap leading-relaxed min-h-[160px]">
                    {result.extracted_text || "(No readable text detected)"}
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/60">
                    <span>Lines Transcribed: {result.lines?.length || 0}</span>
                    <span className="text-sky-400 font-medium">
                      OCR Confidence: {(result.ocr_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Extracted Clinical Entities */}
            <div className="lg:col-span-7">
              <Card className="h-full border-slate-800">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-base text-slate-200">
                        <Pill className="h-4 w-4 text-teal-400" />
                        Structured Clinical Entities
                      </CardTitle>
                      <CardDescription>
                        Click any medicine badge to view its full pharmaceutical monograph
                      </CardDescription>
                    </div>
                    <Badge variant="default" className="text-xs">
                      NER {(result.ner_confidence * 100).toFixed(0)}% Conf
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Medicines */}
                    <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs font-semibold text-slate-400">
                        <span className="flex items-center gap-1.5 text-sky-400">
                          <Pill className="h-3.5 w-3.5" /> MEDICINE
                        </span>
                        <span>{result.entities.medicine.length} detected</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {result.entities.medicine.length > 0 ? (
                          result.entities.medicine.map((med, i) => {
                            const isSelected = selectedMonograph?.name?.toLowerCase().includes(med.toLowerCase());
                            return (
                              <button
                                key={i}
                                onClick={() => {
                                  if (result.monographs && result.monographs[med]) {
                                    setSelectedMonograph(result.monographs[med]);
                                  }
                                }}
                                className={`text-xs font-medium px-2.5 py-0.5 rounded-md border transition-all cursor-pointer ${
                                  isSelected
                                    ? "bg-sky-500 text-white border-sky-400 shadow-md shadow-sky-500/20"
                                    : "bg-sky-500/15 text-sky-300 border-sky-500/30 hover:bg-sky-500/30"
                                }`}
                              >
                                💊 {med}
                              </button>
                            );
                          })
                        ) : (
                          <span className="text-xs text-slate-500 italic">None detected</span>
                        )}
                      </div>
                    </div>

                    {/* Dosages */}
                    <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs font-semibold text-slate-400">
                        <span className="flex items-center gap-1.5 text-emerald-400">
                          <Layers className="h-3.5 w-3.5" /> DOSAGE
                        </span>
                        <span>{result.entities.dosage.length} detected</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {result.entities.dosage.length > 0 ? (
                          result.entities.dosage.map((dos, i) => (
                            <Badge
                              key={i}
                              variant="success"
                              className="text-xs font-medium"
                            >
                              {dos}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-xs text-slate-500 italic">None detected</span>
                        )}
                      </div>
                    </div>

                    {/* Frequency */}
                    <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs font-semibold text-slate-400">
                        <span className="flex items-center gap-1.5 text-amber-400">
                          <Clock className="h-3.5 w-3.5" /> FREQUENCY
                        </span>
                        <span>{result.entities.frequency.length} detected</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {result.entities.frequency.length > 0 ? (
                          result.entities.frequency.map((freq, i) => (
                            <Badge
                              key={i}
                              variant="warning"
                              className="text-xs font-medium"
                            >
                              {freq}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-xs text-slate-500 italic">None detected</span>
                        )}
                      </div>
                    </div>

                    {/* Duration */}
                    <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between text-xs font-semibold text-slate-400">
                        <span className="flex items-center gap-1.5 text-purple-400">
                          <Clock className="h-3.5 w-3.5" /> DURATION
                        </span>
                        <span>{result.entities.duration.length} detected</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {result.entities.duration.length > 0 ? (
                          result.entities.duration.map((dur, i) => (
                            <Badge
                              key={i}
                              variant="secondary"
                              className="text-xs font-medium text-purple-300 border-purple-500/30 bg-purple-500/10"
                            >
                              {dur}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-xs text-slate-500 italic">None detected</span>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Pharmaceutical Monograph Card (RxVision Integration) */}
          {selectedMonograph && (
            <Card className="border-sky-500/40 bg-gradient-to-r from-sky-950/30 via-slate-900/60 to-teal-950/30 shadow-xl">
              <CardHeader className="pb-3 border-b border-sky-900/40">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="h-8 w-8 rounded-lg bg-sky-500/20 border border-sky-500/30 flex items-center justify-center">
                      <BookOpen className="h-4 w-4 text-sky-400" />
                    </div>
                    <div>
                      <CardTitle className="text-base text-slate-100">
                        Drug Monograph: <span className="text-sky-400">{selectedMonograph.name}</span>
                      </CardTitle>
                      <CardDescription className="text-xs text-slate-400">
                        {selectedMonograph.therapeutic_class || "Pharmacological Monograph"} • Generic: {selectedMonograph.generic_name}
                      </CardDescription>
                    </div>
                  </div>
                  <Badge variant="outline" className="text-xs border-sky-500/30 text-sky-300">
                    {selectedMonograph.manufacturer || "FDA Approved"}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="space-y-3">
                  {selectedMonograph.composition && (
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-slate-400 block mb-1">Active Composition:</span>
                      <p className="text-slate-200 leading-relaxed">{selectedMonograph.composition}</p>
                    </div>
                  )}

                  {selectedMonograph.usage && (
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-teal-400 block mb-1">Clinical Indications & Usage:</span>
                      <p className="text-slate-300 leading-relaxed">{selectedMonograph.usage}</p>
                    </div>
                  )}
                </div>

                <div className="space-y-3">
                  {selectedMonograph.standard_dosage && (
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-emerald-400 block mb-1">Standard Clinical Dosing:</span>
                      <p className="text-slate-300 leading-relaxed">{selectedMonograph.standard_dosage}</p>
                    </div>
                  )}

                  {selectedMonograph.precautions && (
                    <div className="p-3 rounded-lg bg-slate-950 border border-amber-900/40 bg-amber-950/20">
                      <span className="font-semibold text-amber-400 block mb-1">Precautions & Warnings:</span>
                      <p className="text-amber-200/90 leading-relaxed">{selectedMonograph.precautions}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Safety & Error Audit Section */}
          <Card className="border-slate-800">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg text-slate-100 flex items-center gap-2">
                    <ShieldAlert className="h-5 w-5 text-sky-400" />
                    Clinical Safety & Error Audit
                  </CardTitle>
                  <CardDescription>
                    Drug-drug interactions, duplicate prescribing, and dosage boundary auditor
                  </CardDescription>
                </div>
                <Badge
                  variant={result.issues.length > 0 ? "warning" : "success"}
                  className="text-xs font-semibold"
                >
                  {result.issues.length} Flagged {result.issues.length === 1 ? "Issue" : "Issues"}
                </Badge>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {/* Disclaimer Notice */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 flex items-center gap-2.5 text-xs text-slate-400">
                <Info className="h-4 w-4 text-sky-400 flex-shrink-0" />
                <span>
                  <strong>Clinical Review Note:</strong> Automated safety checks cross-reference against British Pharmacopoeia, BNF, and FDA monographs. Verify all alerts with the prescribing physician.
                </span>
              </div>

              {/* Issues List */}
              {result.issues.length === 0 ? (
                <div className="p-6 rounded-xl bg-emerald-950/20 border border-emerald-500/30 flex items-center gap-3 text-emerald-300">
                  <CheckCircle2 className="h-6 w-6 text-emerald-400 flex-shrink-0" />
                  <div>
                    <h4 className="font-bold text-sm text-emerald-200">
                      No Prescription Issues Detected
                    </h4>
                    <p className="text-xs text-emerald-400/80 mt-0.5">
                      All detected entities fall within expected therapeutic limits and pass interaction safeguards.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {result.issues.map((iss, idx) => {
                    const isHigh = iss.severity === "HIGH";
                    const isMed = iss.severity === "MEDIUM";

                    const alertVariant = isHigh
                      ? "destructive"
                      : isMed
                      ? "warning"
                      : "info";

                    return (
                      <Alert key={idx} variant={alertVariant} className="space-y-2">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2">
                            <AlertTitle className="text-sm uppercase tracking-wide">
                              {iss.error_type.replace(/_/g, " ")}
                            </AlertTitle>
                          </div>
                          <Badge
                            variant={isHigh ? "destructive" : isMed ? "warning" : "default"}
                            className="text-[10px] font-bold"
                          >
                            {iss.severity} SEVERITY
                          </Badge>
                        </div>

                        <AlertDescription className="text-xs leading-relaxed">
                          {iss.message}
                        </AlertDescription>

                        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300 pt-1 border-t border-white/10">
                          <span className="font-mono text-slate-400">
                            Field: <strong className="text-slate-200">{iss.field}</strong>
                          </span>
                          <span>•</span>
                          <span className="font-mono text-slate-400">
                            Detected: <code className="text-sky-300">{iss.value}</code>
                          </span>
                        </div>

                        {iss.suggestion && (
                          <div className="mt-2 p-2.5 rounded-lg bg-black/30 border border-white/10 flex items-start gap-2 text-xs">
                            <Lightbulb className="h-4 w-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                            <div className="text-slate-300">
                              <strong className="text-emerald-400">Recommendation:</strong>{" "}
                              {iss.suggestion}
                            </div>
                          </div>
                        )}
                      </Alert>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Confidence Scores Meter Card */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* OCR Transcription */}
            {(() => {
              const pct = result.ocr_confidence * 100;
              const level = pct >= 90 ? "clear" : pct >= 70 ? "low" : pct >= 50 ? "medium" : "high";
              const colorVal  = level === "clear" ? "text-emerald-400" : level === "low" ? "text-sky-400" : level === "medium" ? "text-amber-400" : "text-red-400";
              const colorBar  = level === "clear" ? "bg-emerald-500" : level === "low" ? "bg-sky-500" : level === "medium" ? "bg-amber-500" : "bg-red-500";
              const colorBorder = level === "clear" ? "border-emerald-500/30" : level === "low" ? "border-sky-500/30" : level === "medium" ? "border-amber-500/30" : "border-red-500/30";
              const label = level === "clear" ? "Clear (≥90%)" : level === "low" ? "Low Risk (70–90%)" : level === "medium" ? "Medium Risk (50–70%)" : "High Risk (<50%)";
              return (
                <Card className={`p-4 border ${colorBorder} bg-slate-950/60`}>
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="text-slate-400 font-medium">OCR Transcription</span>
                    <span className={`font-bold ${colorVal}`}>{pct.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden mb-2">
                    <div className={`h-full rounded-full transition-all ${colorBar}`} style={{ width: `${pct}%` }} />
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-500">
                    <span className="font-mono">Threshold: 70% / 90%</span>
                    <span className={`font-semibold uppercase tracking-wide ${colorVal}`}>{label}</span>
                  </div>
                  <div className="mt-2 flex gap-1 text-[9px]">
                    <span className="px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/20">&lt;50% HIGH</span>
                    <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/20">50–70% MED</span>
                    <span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/20">70–90% LOW</span>
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/20">≥90% CLEAR</span>
                  </div>
                </Card>
              );
            })()}

            {/* Entity Extraction (NER) */}
            {(() => {
              const pct = result.ner_confidence * 100;
              const level = pct >= 85 ? "clear" : pct >= 70 ? "low" : pct >= 50 ? "medium" : "high";
              const colorVal   = level === "clear" ? "text-emerald-400" : level === "low" ? "text-sky-400" : level === "medium" ? "text-amber-400" : "text-red-400";
              const colorBar   = level === "clear" ? "bg-emerald-500" : level === "low" ? "bg-sky-500" : level === "medium" ? "bg-amber-500" : "bg-red-500";
              const colorBorder = level === "clear" ? "border-emerald-500/30" : level === "low" ? "border-sky-500/30" : level === "medium" ? "border-amber-500/30" : "border-red-500/30";
              const label = level === "clear" ? "Clear (≥85%)" : level === "low" ? "Low Risk (70–85%)" : level === "medium" ? "Medium Risk (50–70%)" : "High Risk (<50%)";
              return (
                <Card className={`p-4 border ${colorBorder} bg-slate-950/60`}>
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="text-slate-400 font-medium">Entity Extraction (NER)</span>
                    <span className={`font-bold ${colorVal}`}>{pct.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden mb-2">
                    <div className={`h-full rounded-full transition-all ${colorBar}`} style={{ width: `${pct}%` }} />
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-500">
                    <span className="font-mono">Threshold: 70% / 85%</span>
                    <span className={`font-semibold uppercase tracking-wide ${colorVal}`}>{label}</span>
                  </div>
                  <div className="mt-2 flex gap-1 text-[9px]">
                    <span className="px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/20">&lt;50% HIGH</span>
                    <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/20">50–70% MED</span>
                    <span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/20">70–85% LOW</span>
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/20">≥85% CLEAR</span>
                  </div>
                </Card>
              );
            })()}

            {/* Safety Rule Confidence */}
            {(() => {
              const highCount = result.issues.filter(i => i.severity === "HIGH").length;
              const medCount  = result.issues.filter(i => i.severity === "MEDIUM").length;
              const hasCritical = result.issues.some(i =>
                ["INTERACTION","CONTRAINDICATION","TOXICITY"].includes(i.error_type?.toUpperCase())
              );
              const rawPct = hasCritical ? 40 : highCount >= 2 ? 42 : highCount === 1 ? 62 : medCount >= 3 ? 62 : medCount >= 1 ? 78 : result.issues.length === 0 ? 98.5 : 91;
              // Safety LOW threshold = 35–85% (user-defined), MEDIUM = 20–35%, HIGH = <20%
              const level = rawPct >= 85 ? "clear" : rawPct >= 35 ? "low" : rawPct >= 20 ? "medium" : "high";
              const colorVal   = level === "clear" ? "text-emerald-400" : level === "low" ? "text-sky-400" : level === "medium" ? "text-amber-400" : "text-red-400";
              const colorBar   = level === "clear" ? "bg-emerald-500" : level === "low" ? "bg-sky-500" : level === "medium" ? "bg-amber-500" : "bg-red-500";
              const colorBorder = level === "clear" ? "border-emerald-500/30" : level === "low" ? "border-sky-500/30" : level === "medium" ? "border-amber-500/30" : "border-red-500/30";
              const label = level === "clear" ? "Clear (≥85%)" : level === "low" ? "Low Risk (35–85%)" : level === "medium" ? "Medium Risk (20–35%)" : "High Risk (<20%)";
              return (
                <Card className={`p-4 border ${colorBorder} bg-slate-950/60`}>
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="text-slate-400 font-medium">Safety Rule Confidence</span>
                    <span className={`font-bold ${colorVal}`}>{rawPct.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden mb-2">
                    <div className={`h-full rounded-full transition-all ${colorBar}`} style={{ width: `${rawPct}%` }} />
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-500">
                    <span className="font-mono">Threshold: 35% / 85%</span>
                    <span className={`font-semibold uppercase tracking-wide ${colorVal}`}>{label}</span>
                  </div>
                  <div className="mt-2 flex gap-1 text-[9px]">
                    <span className="px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/20">&lt;20% HIGH</span>
                    <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/20">20–35% MED</span>
                    <span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/20">35–85% LOW</span>
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/20">≥85% CLEAR</span>
                  </div>
                </Card>
              );
            })()}
          </div>
        </div>
      )}

      {/* Image Cropping & Orientation Modal */}
      <ImageCropperModal
        isOpen={isCropModalOpen}
        imageSrc={rawImageForCrop || ""}
        fileName={selectedFile?.name || "prescription_cropped.png"}
        onCropComplete={handleCropComplete}
        onCancel={handleCropCancel}
      />
    </div>
  );
}

