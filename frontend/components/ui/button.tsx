import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 cursor-pointer select-none",
  {
    variants: {
      variant: {
        default:
          "bg-sky-600 text-white shadow-md hover:bg-sky-500 active:scale-[0.98]",
        teal:
          "bg-teal-600 text-white shadow-md hover:bg-teal-500 active:scale-[0.98]",
        destructive:
          "bg-rose-600 text-white shadow-md hover:bg-rose-500 active:scale-[0.98]",
        outline:
          "border border-slate-700 bg-slate-900/80 hover:bg-slate-800 text-slate-200 hover:text-white shadow-sm",
        secondary:
          "bg-slate-800 text-slate-100 hover:bg-slate-700 active:scale-[0.98]",
        ghost:
          "hover:bg-slate-800/80 text-slate-300 hover:text-white",
        link:
          "text-sky-400 underline-offset-4 hover:underline",
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
