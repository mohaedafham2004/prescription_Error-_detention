# RxAssist — Modern Next.js Frontend

Modern clinical prescription analysis UI built with **Next.js 14+ (App Router)**, **TypeScript**, **Tailwind CSS**, **shadcn/ui**, and **Recharts**.

---

## 1. Prerequisites

Make sure the FastAPI backend is running on `http://localhost:8000`.

---

## 2. Setup & Installation

From the `frontend/` directory:

```bash
# Install npm dependencies
npm install
```

---

## 3. Configuration

Environment variables are stored in `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

To point at a remote deployed backend, update `NEXT_PUBLIC_API_URL`.

---

## 4. Running the Frontend

Start the Next.js development server:

```bash
npm run dev
```

Open [http://localhost:4000](http://localhost:4000) (or [http://127.0.0.1:4000](http://127.0.0.1:4000)) in your browser.

---

## 5. Pages & Features

- **`/` (Prescription Analysis)**:
  - Drag-and-drop prescription scan upload + image preview.
  - "Load Sample Rx" instant pipeline demonstration button.
  - Multi-stage loading progress with animated indicators.
  - Side-by-side OCR transcription and visual document viewer.
  - Structured entity breakdown (Medicine, Dosage, Frequency, Duration) with confidence badges.
  - Safety alerts for drug interactions, out-of-range doses, and duplicate therapy.
  - Export Analysis Report (.json).

- **`/performance` (Model Performance)**:
  - Interactive Recharts bar charts for TrOCR error rates (CER / WER).
  - Multi-class clinical entity extraction charts (Precision, Recall, F1).
  - Informative status card for the Custom Character CNN model.

- **`/about` (About & Architecture)**:
  - 5-step visual pipeline flow cards.
  - Interactive technology stack directory.
  - Hardware constraints and data folder separation principles.
