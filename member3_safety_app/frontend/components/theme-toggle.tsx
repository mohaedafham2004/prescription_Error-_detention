"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Button
        variant="outline"
        size="icon"
        className="h-9 w-9 rounded-xl border border-slate-200 dark:border-[#1E2A44] bg-white dark:bg-[#131B2E] text-slate-400"
        aria-label="Toggle theme"
      >
        <Sun className="h-4 w-4 text-amber-500" />
      </Button>
    );
  }

  const isDark = theme === "dark";

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="h-9 w-9 rounded-xl border border-slate-200 dark:border-[#1E2A44] bg-white dark:bg-[#131B2E] text-slate-700 dark:text-[#E2E8F0] hover:text-[#0284C7] dark:hover:text-[#38BDF8] hover:border-sky-500/50 transition-all shadow-sm group"
      title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
      aria-label="Toggle Theme"
    >
      {isDark ? (
        <Sun className="h-4 w-4 text-[#FBBF24] group-hover:rotate-45 transition-transform" />
      ) : (
        <Moon className="h-4 w-4 text-[#0284C7] group-hover:-rotate-12 transition-transform" />
      )}
    </Button>
  );

}
