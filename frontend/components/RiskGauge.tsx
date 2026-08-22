"use client";

import * as React from "react";
import { AlertTriangle, AlertCircle, CheckCircle2, ShieldAlert, ShieldCheck } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";

export interface RiskData {
  level: "clear" | "low" | "medium" | "high" | string;
  reason?: string;
  message?: string;
}

interface RiskGaugeProps {
  risk: RiskData;
  className?: string;
}

export function RiskGauge({ risk, className = "" }: RiskGaugeProps) {
  const level = (risk.level || "clear").toLowerCase();

  const getAlertVariant = () => {
    switch (level) {
      case "high":
        return "destructive";
      case "medium":
        return "warning";
      case "low":
        return "default";
      case "clear":
      default:
        return "success";
    }
  };

  const getBadgeVariant = () => {
    switch (level) {
      case "high":
        return "destructive";
      case "medium":
        return "warning";
      case "low":
        return "default";
      case "clear":
      default:
        return "success";
    }
  };

  const getIcon = () => {
    switch (level) {
      case "high":
        return <ShieldAlert className="h-5 w-5 text-rose-400" />;
      case "medium":
        return <AlertTriangle className="h-5 w-5 text-amber-400" />;
      case "low":
        return <AlertCircle className="h-5 w-5 text-sky-400" />;
      case "clear":
      default:
        return <ShieldCheck className="h-5 w-5 text-emerald-400" />;
    }
  };

  return (
    <Alert variant={getAlertVariant()} className={`p-5 shadow-xl border ${className}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <AlertTitle className="text-base font-bold tracking-tight flex items-center gap-2">
            {getIcon()}
            <span>OVERALL RISK ASSESSMENT:</span>
            <span className="uppercase text-sky-400">{risk.level}</span>
          </AlertTitle>
          <AlertDescription className="text-sm leading-relaxed text-slate-100 font-medium pt-1">
            {risk.message || "Prescription analysis completed."}
          </AlertDescription>
          {risk.reason && (
            <p className="text-xs text-slate-300/90 italic pt-1">
              Rationale: {risk.reason}
            </p>
          )}
        </div>
        <Badge
          variant={getBadgeVariant()}
          className="text-xs font-bold uppercase tracking-wider px-3 py-1 flex-shrink-0"
        >
          {risk.level} Risk
        </Badge>
      </div>
    </Alert>
  );
}

export default RiskGauge;
