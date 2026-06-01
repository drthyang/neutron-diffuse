# neutron-diffuse

**3D diffuse neutron scattering processing — powder ring removal and 3D-ΔPDF.**

Takes over after instrument data reduction and symmetrization (e.g. from Mantid).

```
[ Symmetrized 3D HKL volume ]
        │
        ▼  DATA PROCESSING
        │  (1) Detect & subtract powder rings  ← isotropic in |Q| → separable
        │  (2) Backfill ring holes             ← interpolate diffuse signal
        │
        ▼  FURTHER ANALYSIS
        │  (3) Punch Bragg peaks
        │  (4) Backfill Bragg holes
        │  (5) 3D-ΔPDF via Fourier transform
        ▼
  [ 3D-ΔPDF in real space ]
```

## Quickstart

```python
import ndiff

# Load symmetrized 3D HKL volume (from Mantid or equivalent)
vol = ndiff.load("experiment_sym.h5")

# ── Data Processing ─────────────────────────────────────────────────────────
# (1) Detect powder rings from radial intensity statistics, subtract profiles
remover = ndiff.preprocessing.PowderRingRemover(snr_mask_threshold=3.0)
vol_sub, rings, I_ring = remover.remove(vol)

# Optionally inspect what was detected
for r in rings:
    print(f"  ring at |Q|={r.q_center:.3f} Å⁻¹, σ={r.q_sigma:.4f} Å⁻¹")

# (2) Fill ring holes by interpolating the surrounding diffuse signal
vol_clean = ndiff.preprocessing.backfill(vol_sub, method="tv", tv_lam=0.08)

# ── Further Analysis ─────────────────────────────────────────────────────────
# (3) Punch Bragg peaks
bragg_keep = ndiff.analysis.bragg_mask(vol_clean, punch_radius_hkl=0.3)
vol_clean.apply_mask(bragg_keep)

# (4) Backfill Bragg holes
vol_diffuse = ndiff.analysis.backfill_bragg(vol_clean)

# (5) 3D-ΔPDF
dpdf = ndiff.analysis.compute_delta_pdf(vol_diffuse, apodization="hann")

# Save
ndiff.save(vol_diffuse, "diffuse_only.h5")
```

## Installation

```bash
git clone https://github.com/thyang-phys/neutron-diffuse
cd neutron-diffuse && pip install -e ".[dev]"
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Algorithm documentation

- [Powder ring removal](docs/algorithms/powder_rings.md)
- [Inpainting methods](docs/algorithms/inpainting.md)

## License

MIT
