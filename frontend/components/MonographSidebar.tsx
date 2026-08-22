"use client";

import * as React from "react";
import { BookOpen, X, Sparkles, AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export interface DrugMonograph {
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

interface MonographSidebarProps {
  monograph: DrugMonograph | null;
  onClose?: () => void;
  className?: string;
}

export function MonographSidebar({ monograph, onClose, className = "" }: MonographSidebarProps) {
  if (!monograph) return null;

  return (
    <Card className={`border-sky-500/30 bg-slate-900/90 shadow-2xl relative overflow-hidden ${className}`}>
      <div className="absolute top-0 right-0 w-32 h-32 bg-sky-500/10 rounded-full blur-2xl pointer-events-none" />
      <CardHeader className="pb-3 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-sky-500/20 border border-sky-500/30 flex items-center justify-center">
              <BookOpen className="h-4 w-4 text-sky-400" />
            </div>
            <div>
              <CardTitle className="text-base text-slate-100">
                Drug Monograph: <span className="text-sky-400">{monograph.name}</span>
              </CardTitle>
              <CardDescription className="text-xs text-slate-400">
                {monograph.therapeutic_class || "Pharmacological Monograph"} • Generic: {monograph.generic_name}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs border-sky-500/30 text-sky-300">
              {monograph.manufacturer || "FDA Approved"}
            </Badge>
            {onClose && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0 text-slate-400 hover:text-slate-200"
                onClick={onClose}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="space-y-3">
          {monograph.composition && (
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="font-semibold text-slate-400 block mb-1">Active Composition:</span>
              <p className="text-slate-200 leading-relaxed">{monograph.composition}</p>
            </div>
          )}

          {monograph.usage && (
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="font-semibold text-teal-400 block mb-1">Clinical Indications & Usage:</span>
              <p className="text-slate-300 leading-relaxed">{monograph.usage}</p>
            </div>
          )}
        </div>

        <div className="space-y-3">
          {monograph.standard_dosage && (
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="font-semibold text-sky-400 block mb-1">Standard Recommended Dosing:</span>
              <p className="text-slate-300 leading-relaxed">{monograph.standard_dosage}</p>
            </div>
          )}

          {monograph.precautions && (
            <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-500/30">
              <span className="font-semibold text-amber-400 block mb-1">Contraindications & Precautions:</span>
              <p className="text-amber-200/90 leading-relaxed">{monograph.precautions}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default MonographSidebar;
