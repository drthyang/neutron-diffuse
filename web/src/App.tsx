import { useState } from "react";

import { DeltaPdfViewer } from "./pages/DeltaPdfViewer";
import { MultiTempViewer } from "./pages/MultiTempViewer";
import { PipelineRunner } from "./pages/PipelineRunner";
import { ReciprocalViewer } from "./pages/ReciprocalViewer";

type Tab = "reciprocal" | "dpdf" | "multi" | "pipeline";

const TABS: { id: Tab; label: string }[] = [
  { id: "reciprocal", label: "Reciprocal cleanup" },
  { id: "dpdf", label: "3D-ΔPDF" },
  { id: "multi", label: "Multi-temperature" },
  { id: "pipeline", label: "Run pipeline" },
];

export function App() {
  const [tab, setTab] = useState<Tab>("reciprocal");
  return (
    <div className="app">
      <header className="app-header">
        <h1>neutron-diffuse</h1>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={t.id === tab ? "active" : ""}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "reciprocal" && <ReciprocalViewer />}
        {tab === "dpdf" && <DeltaPdfViewer />}
        {tab === "multi" && <MultiTempViewer />}
        {tab === "pipeline" && <PipelineRunner />}
      </main>
    </div>
  );
}
