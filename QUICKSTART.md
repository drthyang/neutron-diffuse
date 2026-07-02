# Quickstart

`nebula3d` ships **one** application: a browser console (React UI) that
runs the full cleanup → 3D-ΔPDF → consistency-check workflow. It has two
interchangeable run modes that share the same UI:

| Mode | Use it for | Needs |
| --- | --- | --- |
| **Native** (`nebula3d-web`) | Full-resolution local work; your data in `./data`. | Python 3.10+ |
| **In-browser** (GitHub Pages) | Quick look / sharing; modest volumes; no install. | A browser |

## Native — recommended

Runs the real pipeline locally with no size limit; reads `./data` by default.

```bash
pip install -e ".[web]"
nebula3d-web                 # serves http://127.0.0.1:8000 and opens a browser
```

Useful flags:

```bash
nebula3d-web --data-root /path/to/data   # if raw/ + processed/ live elsewhere
nebula3d-web --no-browser                # headless / remote
```

From a source checkout, build the UI once first (otherwise only the API serves):

```bash
make web-install        # npm install in web/
make ui                 # build the SPA into src/nebula3d/server/static
```

## In-browser — no install

Open the hosted app: **https://drthyang.github.io/nebula3d/**

It runs the real `nebula3d` pipeline entirely in your browser via Pyodide — load
your own `.nxs`/`.h5`, nothing is uploaded. Suited to volumes up to ~50 M
voxels at full float64 precision (a 301×401×401 full-resolution volume fits);
larger data must use the native mode above.

To run that build locally:

```bash
make web-install
cd web && npm run dev:pyodide      # http://localhost:5173
```

## Using the console

Five views in the left sidebar; a session typically runs top to bottom:

1. **Configure** — pick a dataset, tune ring removal / punch / backfill / flatten
   / ΔPDF, then **Run pipeline**. Existing outputs are skipped unless *force* is on.
2. **Reciprocal cleanup** — compare stages (raw → ring-removed → punched →
   backfilled → flattened) on one shared plane and colour scale.
3. **3D-ΔPDF** — three linked real-space orthoslices with a unit-cell overlay.
4. **Consistency check** — inverse-FFT the ΔPDF back to reciprocal space and
   compare *data | back-FFT | residual*, with `|Q|` and real-space `r` bands.
5. **Multi-volume** — DeltaPDF orthoslices side by side across related files.

## Going further

- Scripting / batch CLI recipes: [docs/commands.md](docs/commands.md)
- Web UI reference, architecture, and dev workflow: [docs/web.md](docs/web.md)
- Algorithms and full overview: [README.md](README.md) and [docs/README.md](docs/README.md)
