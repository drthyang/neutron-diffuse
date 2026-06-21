import { useState, type ReactNode } from "react";

import { useHealth } from "./api/hooks";
import { STATIC_MODE } from "./api/staticData";
import {
  BrandGlyph,
  IconCheck,
  IconFlow,
  IconLattice,
  IconLayers,
  IconOrbits,
  IconRun,
} from "./components/ui";
import { ConsistencyViewer } from "./pages/ConsistencyViewer";
import { DeltaPdfViewer } from "./pages/DeltaPdfViewer";
import { MultiTempViewer } from "./pages/MultiTempViewer";
import { PipelineConfig } from "./pages/PipelineConfig";
import { PipelineExecution } from "./pages/PipelineExecution";
import { ReciprocalViewer } from "./pages/ReciprocalViewer";
import { usePipelineStore } from "./state/pipelineStore";

type Tab = "config" | "execution" | "reciprocal" | "dpdf" | "multi" | "consistency";

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
    desc: "Track stage-by-stage progress and the live log for the current pipeline run.",
    icon: <IconRun />,
  },
  {
    id: "reciprocal",
    label: "Reciprocal cleanup",
    desc: "Compare cleanup stages slice-by-slice across the reciprocal-space volume.",
    icon: <IconLattice />,
  },
  {
    id: "dpdf",
    label: "3D-ΔPDF",
    desc: "Linked orthogonal real-space cuts through the difference pair-distribution function.",
    icon: <IconOrbits />,
  },
  {
    id: "consistency",
    label: "Consistency check",
    desc: "Inverse-FFT the ΔPDF back to reciprocal space and compare to the data; band-limit |Q| to separate low- vs high-frequency signal.",
    icon: <IconCheck />,
  },
  {
    id: "multi",
    label: "Multi-temperature",
    desc: "ΔPDF orthoslices side by side across temperatures, with shared cuts and pooled colour scale.",
    icon: <IconLayers />,
  },
];

function renderPage(tab: Tab, setTab: (t: Tab) => void): ReactNode {
  switch (tab) {
    case "config":
      return <PipelineConfig onStarted={() => setTab("execution")} />;
    case "execution":
      return <PipelineExecution />;
    case "reciprocal":
      return <ReciprocalViewer />;
    case "dpdf":
      return <DeltaPdfViewer />;
    case "multi":
      return <MultiTempViewer />;
    case "consistency":
      return <ConsistencyViewer />;
  }
}

// The static (GitHub Pages) build has no backend, so the pipeline pages
// (Configure / Execution) and the stages that need server-side slicing or FFTs
// (Reciprocal cleanup / Consistency check) can't function. Show only the viewers
// that slice the pre-baked ΔPDF volumes client-side.
const STATIC_TABS: Tab[] = ["dpdf", "multi"];
const VISIBLE_NAV = STATIC_MODE ? NAV.filter((n) => STATIC_TABS.includes(n.id)) : NAV;

export function App() {
  const [tab, setTab] = useState<Tab>(STATIC_MODE ? "dpdf" : "config");
  const health = useHealth();
  const apiUp = health.isSuccess;
  const running = usePipelineStore((s) => s.running);
  const active = VISIBLE_NAV.find((n) => n.id === tab) ?? VISIBLE_NAV[0];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-glyph">
            <BrandGlyph />
          </span>
          <span className="brand-name">
            <b>neutron-diffuse</b>
            <span className="brand-sub">scattering console</span>
          </span>
        </div>

        <nav className="nav">
          {VISIBLE_NAV.map((n) => (
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
            {apiUp ? "API connected" : "API offline"}
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
