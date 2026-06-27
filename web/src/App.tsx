import { useState, type ReactNode } from "react";

import { useHealth } from "./api/hooks";
import { PYODIDE_MODE } from "./api/pyodideEngine";
import {
  BrandGlyph,
  IconFlow,
  IconLattice,
  IconLayers,
  IconOrbits,
  IconProfileWave,
  IconRun,
  IconTransform,
} from "./components/ui";
import { ConsistencyViewer } from "./pages/ConsistencyViewer";
import { BraggProfileViewer } from "./pages/BraggProfileViewer";
import { DeltaPdfViewer } from "./pages/DeltaPdfViewer";
import { MultiTempViewer } from "./pages/MultiTempViewer";
import { PipelineConfig } from "./pages/PipelineConfig";
import { PipelineExecution } from "./pages/PipelineExecution";
import { ReciprocalViewer } from "./pages/ReciprocalViewer";
import { usePipelineStore } from "./state/pipelineStore";

export type Tab = "config" | "execution" | "reciprocal" | "bragg" | "dpdf" | "multi" | "consistency";

const NAV: { id: Tab; label: string; desc: string; icon: ReactNode }[] = [
  {
    id: "config",
    label: "Configure",
    desc: "Set parameters for the full reduction from raw volume to consistency-checked 3D-ΔPDF, then launch a run.",
    icon: <IconFlow />,
  },
  {
    id: "execution",
    label: "Execution",
    desc: "Live progress for the running reduction — per-stage status and the streaming event log.",
    icon: <IconRun />,
  },
  {
    id: "reciprocal",
    label: "Reciprocal cleanup",
    desc: "Compare cleanup stages slice-by-slice across the reciprocal-space volume.",
    icon: <IconLattice />,
  },
  {
    id: "bragg",
    label: "Bragg profile",
    desc: "Fitted radius vs. pad-free measured width for every punched peak across all three reciprocal axes — peaks sagging to the half-voxel floor are resolution-limited.",
    icon: <IconProfileWave />,
  },
  {
    id: "dpdf",
    label: "3D-ΔPDF",
    desc: "Linked orthogonal real-space cuts through the difference pair-distribution function.",
    icon: <IconOrbits />,
  },
  {
    id: "consistency",
    label: "Q–R Band Transform",
    desc: "Inverse-FFT the ΔPDF back to reciprocal space and compare to the data; band-limit |Q| to separate low- vs high-frequency signal.",
    icon: <IconTransform />,
  },
  {
    id: "multi",
    label: "Multi-volume",
    desc: "ΔPDF orthoslices side by side across related files, with shared cuts and pooled colour scale.",
    icon: <IconLayers />,
  },
];

function renderPage(tab: Tab, setTab: (t: Tab) => void): ReactNode {
  switch (tab) {
    case "config":
      return <PipelineConfig onStarted={() => setTab("execution")} />;
    case "execution":
      return <PipelineExecution onNavigate={setTab} />;
    case "reciprocal":
      return <ReciprocalViewer />;
    case "bragg":
      return <BraggProfileViewer />;
    case "dpdf":
      return <DeltaPdfViewer />;
    case "multi":
      return <MultiTempViewer />;
    case "consistency":
      return <ConsistencyViewer />;
  }
}

export function App() {
  const [tab, setTab] = useState<Tab>("config");
  const health = useHealth();
  const apiUp = health.isSuccess;
  const running = usePipelineStore((s) => s.running);
  const active = NAV.find((n) => n.id === tab) ?? NAV[0];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-glyph">
            <BrandGlyph />
          </span>
          <span className="brand-name">
            <b>nebula3d</b>
            <span className="brand-sub">scattering console</span>
          </span>
        </div>

        <nav className="nav">
          {NAV.map((n) => (
            <button
              key={n.id}
              type="button"
              className={n.id === tab ? "active" : ""}
              onClick={() => setTab(n.id)}
            >
              {n.icon}
              {n.label}
              {n.id === "execution" && running && (
                <span className="nav-dot" title="a job is running" />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <span className="api-status">
            <span className={`api-dot ${apiUp ? "ok" : "down"}`} />
            {PYODIDE_MODE
              ? "in-browser engine"
              : apiUp
                ? "API connected"
                : "API offline"}
          </span>
          <span className="ver">
            <span className="ver-num">v0.2.0</span>
            <span className="ver-tag">alpha</span>
          </span>
          <span className="copyright">© 2026 Tsung-Han Yang</span>
        </div>
      </aside>

      <main className="main">
        <header className="page-head">
          <h2>{active.label}</h2>
          <p>{active.desc}</p>
        </header>
        {renderPage(tab, setTab)}
      </main>
    </div>
  );
}
