# neutron-diffuse

**3D diffuse neutron scattering data processing — aluminum background removal and inpainting.**

Neutron single-crystal diffuse scattering experiments are contaminated by powder rings from aluminum (sample environment, cryostats). This toolkit provides:

- **Precise Al-ring detection** in 3D reciprocal space given any UB matrix
- **Robust adaptive masking** with minimal boundary artifacts
- **Advanced inpainting** to fill masked regions using symmetry, interpolation, and variational methods
- **Uncertainty quantification** for filled voxels

## Quickstart

```python
import ndiff

# Load 3D HKL volume
vol = ndiff.load("experiment.h5")

# Detect and mask Al powder rings
mask = ndiff.background.aluminum_mask(vol, al_lattice=4.046)

# Fill masked voxels
filled = ndiff.inpainting.fill(vol, mask, method="symmetry+tv")

# Export
ndiff.save(filled, "experiment_cleaned.h5")
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

## Documentation

See [docs/algorithms/](docs/algorithms/) for detailed method descriptions.

## License

MIT
