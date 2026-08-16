import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";

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
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-screen bg-[#070b12] text-slate-100 flex flex-col selection:bg-sky-500/30 selection:text-sky-200">
        <div className="fixed inset-0 bg-radial-glow pointer-events-none z-0" />
        <Navbar />
        <main className="flex-1 z-10">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
