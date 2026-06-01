# neutron-diffuse

**3D diffuse neutron scattering processing — from reduced HKL data to 3D-ΔPDF.**

Takes over after instrument data reduction and provides a two-stage pipeline:

```
[ Reduced 3D HKL volume ]
        │
        ▼  DATA PROCESSING
        │  (1) Symmetrize
        │  (2) Remove Al signals
        │  (3) Backfill Al holes
        │
        ▼  FURTHER ANALYSIS
        │  (4) Remove Bragg peaks  (punch-and-fill)
        │  (5) Backfill Bragg holes
        │  (6) 3D-ΔPDF via Fourier transform
        ▼
  [ 3D-ΔPDF in real space ]
```

## Quickstart

```python
import ndiff

# Load reduced 3D HKL volume
vol = ndiff.load("experiment.h5")

# ── Data Processing ─────────────────────────────────────────────
# (1) Symmetrize
vol_sym, outliers = ndiff.preprocessing.symmetrize(vol, laue_class="m3m")

# (2) Remove Al signals
al_keep = ndiff.preprocessing.aluminum_mask(vol_sym, al_lattice=4.046)
vol_sym.apply_mask(al_keep)

# (3) Backfill Al holes
vol_clean = ndiff.preprocessing.backfill_al(vol_sym)

# ── Further Analysis ─────────────────────────────────────────────
# (4) Remove Bragg peaks
bragg_keep = ndiff.analysis.bragg_mask(vol_clean, punch_radius_hkl=0.3)
vol_clean.apply_mask(bragg_keep)

# (5) Backfill Bragg holes
vol_diffuse = ndiff.analysis.backfill_bragg(vol_clean)

# (6) 3D-ΔPDF
dpdf = ndiff.analysis.compute_delta_pdf(vol_diffuse, apodization="hann")

# Save results
ndiff.save(vol_diffuse, "diffuse_only.h5")
```

## Installation

```bash
pip install neutron-diffuse        # once on PyPI
# or from source:
git clone https://github.com/thyang-phys/neutron-diffuse
cd neutron-diffuse && pip install -e ".[dev]"
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the phased development plan.

## Algorithm documentation

- [Al removal](docs/algorithms/al_removal.md)
- [Inpainting methods](docs/algorithms/inpainting.md)

## License

MIT
