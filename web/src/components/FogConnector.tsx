// Decorative connector between the Q–R flow panels: soft blurred "nebula" puffs
// and sparkle particles drift left→right to convey the transform direction
// (Q→R, R→Q′) without an explicit arrow — and to echo the app name "Nebula".
//
// Purely visual (aria-hidden).  Positions are expressed relative to the channel
// height (percent) rather than the design prototype's fixed pixel offsets, so the
// fog fills the gap at any panel height.  Particle parameters are randomised once
// per mount (useMemo) so they stay put across re-renders.

import { useMemo } from "react";

const PUFF_TINTS = [
  "rgba(145, 165, 255, 0.21)", // nebula blue
  "rgba(125, 155, 255, 0.20)",
  "rgba(155, 135, 255, 0.18)", // violet
  "rgba(180, 145, 255, 0.16)",
  "rgba(120, 195, 235, 0.16)", // teal
];
const SPARKLE_TINTS = [
  "rgba(160, 210, 240, 0.85)",
  "rgba(176, 200, 255, 0.90)",
  "rgba(190, 175, 255, 0.85)",
  "rgba(214, 228, 255, 0.95)",
  "rgba(236, 242, 255, 0.98)",
];

const rand = (min: number, max: number) => min + Math.random() * (max - min);

export function FogConnector() {
  const { puffs, sparkles } = useMemo(() => {
    const puffs = Array.from({ length: 10 }, (_, i) => ({
      key: i,
      top: rand(6, 76), // % of channel height
      height: rand(28, 52), // % of channel height — tall blurred ellipses
      width: rand(32, 56), // px (channel is ~50px; wider puffs clip into a soft fog)
      tint: PUFF_TINTS[i % PUFF_TINTS.length],
      blur: rand(7, 13),
      duration: rand(6.5, 13),
      delay: -rand(0, 12),
    }));
    const sparkles = Array.from({ length: 14 }, (_, i) => ({
      key: i,
      top: rand(5, 93),
      size: rand(0.8, 1.7),
      tint: SPARKLE_TINTS[i % SPARKLE_TINTS.length],
      duration: rand(4.5, 8),
      delay: -rand(0, 7),
    }));
    return { puffs, sparkles };
  }, []);

  return (
    <div className="qr-fog" aria-hidden="true">
      {puffs.map((p) => (
        <span
          key={`p${p.key}`}
          className="qr-fog-puff"
          style={{
            top: `${p.top}%`,
            height: `${p.height}%`,
            width: `${p.width}px`,
            background: `radial-gradient(ellipse, ${p.tint}, transparent 68%)`,
            filter: `blur(${p.blur}px)`,
            animationDuration: `${p.duration}s`,
            animationDelay: `${p.delay}s`,
          }}
        />
      ))}
      {sparkles.map((s) => (
        <span
          key={`s${s.key}`}
          className="qr-fog-spark"
          style={{
            top: `${s.top}%`,
            width: `${s.size}px`,
            height: `${s.size}px`,
            background: s.tint,
            animationDuration: `${s.duration}s`,
            animationDelay: `${s.delay}s`,
          }}
        />
      ))}
    </div>
  );
}
