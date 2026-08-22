import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 cursor-pointer select-none",
  {
    variants: {
      variant: {
        default:
          "bg-[#0284C7] hover:bg-[#0369A1] text-white dark:bg-[#38BDF8] dark:hover:bg-[#0EA5E9] dark:text-[#0B1120] font-semibold shadow-md active:scale-[0.98]",
        teal:
          "bg-[#059669] hover:bg-[#047857] text-white dark:bg-[#34D399] dark:hover:bg-[#10B981] dark:text-[#0B1120] font-semibold shadow-md active:scale-[0.98]",
        destructive:
          "bg-[#DC2626] hover:bg-[#B91C1C] text-white dark:bg-[#F87171] dark:hover:bg-[#EF4444] dark:text-white font-semibold shadow-md active:scale-[0.98]",
        outline:
          "border border-slate-200 dark:border-[#1E2A44] bg-white dark:bg-[#131B2E] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-[#1A243B] shadow-sm",
        secondary:
          "bg-slate-100 dark:bg-[#1E2A44] text-slate-800 dark:text-slate-100 hover:bg-slate-200 dark:hover:bg-[#2A3B5E] active:scale-[0.98]",
        ghost:
          "hover:bg-slate-100 dark:hover:bg-[#1E2A44] text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white",
        link:
          "text-[#0284C7] dark:text-[#38BDF8] underline-offset-4 hover:underline",

      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-11 rounded-lg px-8 text-base",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
