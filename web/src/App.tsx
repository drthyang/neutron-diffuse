import { useState, type ReactNode } from "react";

import { useHealth } from "./api/hooks";
import {
  BrandGlyph,
  IconFlow,
  IconLattice,
  IconLayers,
  IconOrbits,
} from "./components/ui";
import { DeltaPdfViewer } from "./pages/DeltaPdfViewer";
import { MultiTempViewer } from "./pages/MultiTempViewer";
import { PipelineRunner } from "./pages/PipelineRunner";
import { ReciprocalViewer } from "./pages/ReciprocalViewer";

type Tab = "reciprocal" | "dpdf" | "multi" | "pipeline";

const NAV: {
  id: Tab;
  label: string;
  desc: string;
  icon: ReactNode;
  page: ReactNode;
}[] = [
  {
    id: "pipeline",
    label: "Run pipeline",
    desc: "Configure and execute the five-stage reduction from raw volume to 3D-ΔPDF.",
    icon: <IconFlow />,
    page: <PipelineRunner />,
  },
  {
    id: "reciprocal",
    label: "Reciprocal cleanup",
    desc: "Compare cleanup stages slice-by-slice across the reciprocal-space volume.",
    icon: <IconLattice />,
    page: <ReciprocalViewer />,
  },
  {
    id: "dpdf",
    label: "3D-ΔPDF",
    desc: "Linked orthogonal real-space cuts through the difference pair-distribution function.",
    icon: <IconOrbits />,
    page: <DeltaPdfViewer />,
  },
  {
    id: "multi",
    label: "Multi-temperature",
    desc: "ΔPDF orthoslices side by side across temperatures, with shared cuts and pooled colour scale.",
    icon: <IconLayers />,
    page: <MultiTempViewer />,
  },
];

export function App() {
  const [tab, setTab] = useState<Tab>("pipeline");
  const health = useHealth();
  const apiUp = health.isSuccess;
  const active = NAV.find((n) => n.id === tab) ?? NAV[0];

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
          {NAV.map((n) => (
            <button
              key={n.id}
              type="button"
              className={n.id === tab ? "active" : ""}
              onClick={() => setTab(n.id)}
            >
              {n.icon}
              {n.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <span className="api-status">
            <span className={`api-dot ${apiUp ? "ok" : "down"}`} />
            {apiUp ? "API connected" : "API offline"}
          </span>
          <span className="ver">
            <span className="ver-num">v0.1.0</span>
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
        {active.page}
      </main>
    </div>
  );
}
