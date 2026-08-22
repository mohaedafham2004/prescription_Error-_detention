import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "bg-sky-500/15 text-[#0284C7] dark:text-[#38BDF8] border-sky-200 dark:border-sky-500/30",
        secondary:
          "bg-slate-100 dark:bg-slate-800/80 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-[#1E2A44]",
        destructive:
          "bg-red-50 dark:bg-red-500/15 text-[#DC2626] dark:text-[#F87171] border-red-200 dark:border-red-500/30 font-semibold",
        warning:
          "bg-amber-50 dark:bg-amber-500/15 text-[#D97706] dark:text-[#FBBF24] border-amber-200 dark:border-amber-500/30 font-semibold",
        success:
          "bg-emerald-50 dark:bg-emerald-500/15 text-[#059669] dark:text-[#34D399] border-emerald-200 dark:border-emerald-500/30 font-semibold",
        outline:
          "text-slate-600 dark:text-slate-300 border-slate-200 dark:border-[#1E2A44] bg-white dark:bg-[#131B2E]",

      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
