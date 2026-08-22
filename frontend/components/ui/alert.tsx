import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative w-full rounded-xl border p-4 [&>svg~*]:pl-8 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground backdrop-blur-md shadow-lg",
  {
    variants: {
      variant: {
        default:
          "border-slate-200 dark:border-[#1E2A44] bg-white dark:bg-[#131B2E] text-slate-800 dark:text-slate-100",
        info:
          "border-sky-200 dark:border-sky-500/40 bg-sky-50/80 dark:bg-[#0C4A6E]/25 text-[#0369A1] dark:text-[#BAE6FD] [&>svg]:text-[#0284C7] dark:[&>svg]:text-[#38BDF8]",
        warning:
          "border-amber-200 dark:border-amber-500/40 bg-amber-50/80 dark:bg-[#78350F]/25 text-[#B45309] dark:text-[#FDE68A] [&>svg]:text-[#D97706] dark:[&>svg]:text-[#FBBF24]",
        destructive:
          "border-red-200 dark:border-red-500/40 bg-red-50/80 dark:bg-[#7F1D1D]/25 text-[#B91C1C] dark:text-[#FECACA] [&>svg]:text-[#DC2626] dark:[&>svg]:text-[#F87171]",
        success:
          "border-emerald-200 dark:border-emerald-500/40 bg-emerald-50/80 dark:bg-[#064E3B]/25 text-[#15803D] dark:text-[#A7F3D0] [&>svg]:text-[#059669] dark:[&>svg]:text-[#34D399]",

      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const Alert = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div
    ref={ref}
    role="alert"
    className={cn(alertVariants({ variant }), className)}
    {...props}
  />
));
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn("mb-1 font-bold leading-none tracking-tight text-base flex items-center gap-2", className)}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm [&_p]:leading-relaxed opacity-90", className)}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
