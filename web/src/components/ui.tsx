// Shared UI primitives for the console: labelled fields, sliders with value
// readouts, switches, segmented controls, empty states, and the inline icon set.

import { useEffect, useRef, useState, type ReactNode } from "react";

/* ----------------------------------------------------------------- fields */

export function Field({
  label,
  children,
  grow = false,
}: {
  label: ReactNode;
  children: ReactNode;
  grow?: boolean;
}) {
  return (
    <div className={`field${grow ? " grow" : ""}`}>
      <span className="field-label">{label}</span>
      {children}
    </div>
  );
}

// An editable readout: type a value and commit (Enter / blur) to snap to the
// nearest data point.  Renders in place of the plain `readout` text.
export interface ValueInputConfig {
  value: number; // the real-space value shown when not being edited
  onCommit: (v: number) => void; // caller snaps to the closest data point
  prefix?: string; // e.g. "H ="
  suffix?: string; // e.g. "r.l.u."
}

function fmtRlu(v: number): string {
  return Number(v.toFixed(4)).toString();
}

function ValueInput({
  value,
  onCommit,
  prefix,
  suffix,
  disabled,
}: ValueInputConfig & { disabled?: boolean }) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const v = Number(draft);
    if (draft.trim() !== "" && Number.isFinite(v)) onCommit(v);
    setDraft(null);
  };
  return (
    <span className="readout-edit">
      {prefix && <span>{prefix}</span>}
      <input
        type="number"
        className="value-input"
        value={draft ?? fmtRlu(value)}
        disabled={disabled}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          else if (e.key === "Escape") setDraft(null);
        }}
        onBlur={commit}
      />
      {suffix && <span>{suffix}</span>}
    </span>
  );
}

export function Slider({
  label,
  readout,
  valueInput,
  value,
  min,
  max,
  step = 1,
  disabled = false,
  grow = false,
  onChange,
}: {
  label: ReactNode;
  readout?: string;
  valueInput?: ValueInputConfig;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  grow?: boolean;
  onChange: (v: number) => void;
}) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <div className={`field${grow ? " grow" : ""}`}>
      <div className="field-row">
        <span className="field-label">{label}</span>
        {valueInput ? (
          <ValueInput {...valueInput} disabled={disabled} />
        ) : readout !== undefined ? (
          <span className="readout">{readout}</span>
        ) : null}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={{ "--p": `${pct}%` } as React.CSSProperties}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function Switch({
  label,
  checked,
  onChange,
}: {
  label: ReactNode;
  checked: boolean;
  onChange: (b: boolean) => void;
}) {
  return (
    <label className="switch">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="switch-track" />
      <span className="switch-label">{label}</span>
    </label>
  );
}

export function Segmented({
  options,
  value,
  onChange,
}: {
  options: readonly string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="segmented" role="group">
      {options.map((o) => (
        <button
          key={o}
          type="button"
          className={o === value ? "on" : ""}
          onClick={() => onChange(o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

/* Renders a colormap LUT as a thin gradient strip. */
export function ColormapBar({ lut }: { lut: Uint8ClampedArray }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = 256;
    canvas.height = 1;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(256, 1);
    img.data.set(lut);
    ctx.putImageData(img, 0, 0);
  }, [lut]);
  return <canvas ref={ref} className="cmap-bar" />;
}

/* ------------------------------------------------------------------ states */

export function EmptyState({
  title,
  hint,
  error = false,
  icon,
}: {
  title: string;
  hint?: string;
  error?: boolean;
  icon?: ReactNode;
}) {
  return (
    <div className={`empty${error ? " error" : ""}`}>
      <span className="empty-icon">{icon ?? <IconLattice size={20} />}</span>
      <span className="empty-title">{title}</span>
      {hint && <span className="empty-hint">{hint}</span>}
    </div>
  );
}

export function MetaStrip({
  items,
}: {
  items: { key: string; value: ReactNode }[];
}) {
  return (
    <div className="meta-strip">
      {items.map((it) => (
        <div key={it.key} className="meta-item">
          <span className="meta-key">{it.key}</span>
          <span className="meta-val">{it.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------- icons */

interface IconProps {
  size?: number;
}

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

/* 3×3 reciprocal-lattice dots */
export function IconLattice({ size = 17 }: IconProps) {
  const p = [3.5, 9, 14.5];
  return (
    <svg width={size} height={size} viewBox="0 0 18 18">
      {p.flatMap((y) =>
        p.map((x) => (
          <circle
            key={`${x}-${y}`}
            cx={x}
            cy={y}
            r={x === 9 && y === 9 ? 2 : 1.3}
            fill="currentColor"
          />
        )),
      )}
    </svg>
  );
}

/* concentric PDF shells */
export function IconOrbits({ size = 17 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" {...stroke}>
      <circle cx={9} cy={9} r={7} />
      <circle cx={9} cy={9} r={3.6} />
      <circle cx={9} cy={9} r={0.8} fill="currentColor" stroke="none" />
    </svg>
  );
}

/* stacked temperature layers */
export function IconLayers({ size = 17 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" {...stroke}>
      <path d="M9 2.5 16 6 9 9.5 2 6Z" />
      <path d="M3.5 9.25 2 10l7 3.5 7-3.5-1.5-.75" />
      <path d="M3.5 13.25 2 14l7 3.5 7-3.5-1.5-.75" opacity={0.55} />
    </svg>
  );
}

/* pipeline flow: nodes joined by edges */
export function IconFlow({ size = 17 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" {...stroke}>
      <circle cx={3.5} cy={9} r={2} />
      <circle cx={14.5} cy={3.5} r={2} />
      <circle cx={14.5} cy={14.5} r={2} />
      <path d="M5.4 8.1 12.6 4.4M5.4 9.9l7.2 3.7" />
    </svg>
  );
}

/* scattering brand glyph: centre beam + diffuse satellites */
export function BrandGlyph({ size = 18 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18">
      <circle cx={9} cy={9} r={2.4} fill="currentColor" />
      <g fill="currentColor" opacity={0.55}>
        <circle cx={9} cy={2.8} r={1.2} />
        <circle cx={9} cy={15.2} r={1.2} />
        <circle cx={2.8} cy={9} r={1.2} />
        <circle cx={15.2} cy={9} r={1.2} />
        <circle cx={4.6} cy={4.6} r={0.9} />
        <circle cx={13.4} cy={4.6} r={0.9} />
        <circle cx={4.6} cy={13.4} r={0.9} />
        <circle cx={13.4} cy={13.4} r={0.9} />
      </g>
    </svg>
  );
}

export function IconAlert({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" {...stroke}>
      <path d="M9 2.5 16.5 15.5H1.5Z" />
      <path d="M9 7.5v3.5" />
      <circle cx={9} cy={13.2} r={0.7} fill="currentColor" stroke="none" />
    </svg>
  );
}
