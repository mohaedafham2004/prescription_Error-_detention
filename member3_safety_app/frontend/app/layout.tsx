import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { ThemeProvider } from "@/components/theme-provider";

export const metadata: Metadata = {
  title: "RxAssist · Smart Prescription Error Detection NLP",
  description:
    "AI-assisted clinical prescription analysis pipeline: handwriting OCR transcription, clinical NER entity extraction, and automated drug safety verification.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full antialiased">
      <body
        suppressHydrationWarning
        className="min-h-screen flex flex-col selection:bg-sky-500/30 selection:text-sky-200"
      >
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          <div className="fixed inset-0 bg-radial-glow pointer-events-none z-0" />
          <Navbar />
          <main className="flex-1 z-10">{children}</main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>

  );
}

