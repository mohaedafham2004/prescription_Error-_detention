"use client";

import * as React from "react";
import ReactCrop, { type Crop, type PixelCrop } from "react-image-crop";
import "react-image-crop/dist/ReactCrop.css";
import {
  Crop as CropIcon,
  RotateCw,
  Check,
  X,
  Maximize2,
  Minimize2,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface ImageCropperModalProps {
  isOpen: boolean;
  imageSrc: string;
  fileName?: string;
  onCropComplete: (file: File, previewUrl: string) => void;
  onCancel: () => void;
}

export function ImageCropperModal({
  isOpen,
  imageSrc,
  fileName = "prescription_cropped.png",
  onCropComplete,
  onCancel,
}: ImageCropperModalProps) {
  const [crop, setCrop] = React.useState<Crop>({
    unit: "%",
    width: 90,
    height: 90,
    x: 5,
    y: 5,
  });
  const [completedCrop, setCompletedCrop] = React.useState<PixelCrop | null>(null);
  const [rotation, setRotation] = React.useState<number>(0);
  const [isProcessing, setIsProcessing] = React.useState<boolean>(false);
  const imgRef = React.useRef<HTMLImageElement>(null);

  // Reset when a new image is loaded
  React.useEffect(() => {
    if (isOpen) {
      setRotation(0);
      setCrop({
        unit: "%",
        width: 90,
        height: 90,
        x: 5,
        y: 5,
      });
      setCompletedCrop(null);
    }
  }, [isOpen, imageSrc]);

  if (!isOpen || !imageSrc) return null;

  const handleRotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  const handleResetCrop = () => {
    setCrop({
      unit: "%",
      width: 96,
      height: 96,
      x: 2,
      y: 2,
    });
  };

  const handleApplyCrop = async () => {
    const image = imgRef.current;
    if (!image) return;

    setIsProcessing(true);

    try {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not create canvas context");

      const naturalWidth = image.naturalWidth;
      const naturalHeight = image.naturalHeight;

      // Handle rotation transformation
      const isRotated90or270 = rotation === 90 || rotation === 270;
      const rotatedCanvas = document.createElement("canvas");
      const rCtx = rotatedCanvas.getContext("2d");
      if (!rCtx) throw new Error("Could not create rotated canvas");

      rotatedCanvas.width = isRotated90or270 ? naturalHeight : naturalWidth;
      rotatedCanvas.height = isRotated90or270 ? naturalWidth : naturalHeight;

      rCtx.translate(rotatedCanvas.width / 2, rotatedCanvas.height / 2);
      rCtx.rotate((rotation * Math.PI) / 180);
      rCtx.drawImage(image, -naturalWidth / 2, -naturalHeight / 2);

      // Crop coordinates calculation
      let cropX = 0;
      let cropY = 0;
      let cropW = rotatedCanvas.width;
      let cropH = rotatedCanvas.height;

      if (completedCrop && completedCrop.width > 0 && completedCrop.height > 0) {
        const scaleX = rotatedCanvas.width / image.width;
        const scaleY = rotatedCanvas.height / image.height;
        cropX = completedCrop.x * scaleX;
        cropY = completedCrop.y * scaleY;
        cropW = completedCrop.width * scaleX;
        cropH = completedCrop.height * scaleY;
      }

      canvas.width = cropW;
      canvas.height = cropH;

      ctx.drawImage(
        rotatedCanvas,
        cropX,
        cropY,
        cropW,
        cropH,
        0,
        0,
        cropW,
        cropH
      );

      // Convert canvas to Blob & File
      canvas.toBlob((blob) => {
        if (!blob) {
          setIsProcessing(false);
          return;
        }
        const croppedFile = new File([blob], fileName, { type: "image/png" });
        const croppedUrl = URL.createObjectURL(blob);
        setIsProcessing(false);
        onCropComplete(croppedFile, croppedUrl);
      }, "image/png", 0.98);
    } catch (err) {
      console.error("Error cropping image:", err);
      setIsProcessing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="bg-white dark:bg-[#131B2E] border border-slate-200 dark:border-[#1E2A44] rounded-2xl max-w-4xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-[#1E2A44] bg-slate-50 dark:bg-[#0B1120]/80">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-[#0284C7] dark:text-[#38BDF8]">
              <CropIcon className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-slate-900 dark:text-[#E2E8F0] text-sm">
                  Crop & Frame Prescription
                </h3>
                <Badge variant="outline" className="text-[10px] py-0 border-slate-200 dark:border-[#1E2A44] text-slate-500 dark:text-[#94A3B8]">
                  Pre-OCR Optimization
                </Badge>
              </div>
              <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
                Drag corners to focus on handwritten medicine and dosage lines for highest OCR accuracy.
              </p>
            </div>
          </div>

          <button
            onClick={onCancel}
            className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 p-1.5 rounded-lg hover:bg-slate-200/60 dark:hover:bg-[#1E2A44] transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Interactive Cropper Canvas Area */}
        <div className="flex-1 overflow-auto p-6 flex items-center justify-center bg-slate-100 dark:bg-[#0B1120]/90 min-h-[360px]">
          <div className="relative max-h-[60vh] flex items-center justify-center">
            <ReactCrop
              crop={crop}
              onChange={(c) => setCrop(c)}
              onComplete={(c) => setCompletedCrop(c)}
              className="max-h-[58vh] max-w-full rounded-lg overflow-hidden border border-slate-300 dark:border-[#1E2A44]"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                ref={imgRef}
                src={imageSrc}
                alt="Prescription to crop"
                style={{
                  transform: `rotate(${rotation}deg)`,
                  maxHeight: "58vh",
                  maxWidth: "100%",
                  objectFit: "contain",
                }}
                className="transition-transform duration-200"
              />
            </ReactCrop>
          </div>
        </div>

        {/* Toolbar & Actions Footer */}
        <div className="px-6 py-4 border-t border-slate-200 dark:border-[#1E2A44] bg-slate-50 dark:bg-[#0B1120]/80 flex flex-wrap items-center justify-between gap-3">
          {/* Transform Controls */}
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleRotate}
              className="text-xs gap-1.5 border-slate-200 dark:border-[#1E2A44] text-slate-700 dark:text-[#E2E8F0] hover:bg-slate-100 dark:hover:bg-[#1E2A44]"
            >
              <RotateCw className="h-3.5 w-3.5" />
              Rotate 90° {rotation > 0 && `(${rotation}°)`}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleResetCrop}
              className="text-xs gap-1.5 border-slate-200 dark:border-[#1E2A44] text-slate-700 dark:text-[#E2E8F0] hover:bg-slate-100 dark:hover:bg-[#1E2A44]"
            >
              <Maximize2 className="h-3.5 w-3.5" />
              Full Frame
            </Button>
          </div>

          {/* Action CTAs */}
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onCancel}
              className="text-xs text-slate-500 dark:text-[#94A3B8] hover:text-slate-800 dark:hover:text-white"
            >
              Use Uncropped
            </Button>
            <Button
              type="button"
              variant="default"
              size="sm"
              onClick={handleApplyCrop}
              disabled={isProcessing}
              className="text-xs font-bold gap-1.5 bg-[#0284C7] dark:bg-[#38BDF8] hover:bg-[#0369A1] dark:hover:bg-[#0EA5E9] text-white dark:text-[#0B1120] shadow-md"
            >
              <Check className="h-3.5 w-3.5" />
              {isProcessing ? "Processing..." : "Apply Crop & Continue"}
            </Button>
          </div>
        </div>
      </div>
    </div>

  );
}
