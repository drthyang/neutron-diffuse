# neutron-diffuse web

The browser console for `neutron-diffuse`: a React + TypeScript single-page app
(Vite) over the FastAPI backend in `src/ndiff/server/`. It runs the reduction
pipeline and unifies the cleanup and 3D-ΔPDF viewers.

## Develop

```bash
# backend — API on :8000
ndiff-web --no-browser --reload

# this app — Vite dev server on :5173 (proxies /api to :8000)
npm install
npm run dev            # http://localhost:5173
```

## Checks & build

```bash
npm run lint           # eslint
npm run typecheck      # tsc --noEmit
npm run build          # tsc + vite build → ../src/ndiff/server/static (bundled in the wheel)
```

## Layout

| Path | What |
| --- | --- |
| `src/App.tsx` | sidebar shell + view routing |
| `src/pages/` | one component per sidebar view (pipeline, reciprocal, ΔPDF, multi-temp) |
| `src/components/` | shared panels (`SliceCanvas`, `SlicePanel`, `DpdfPanel`, `UnitCellGrid`) and UI primitives (`ui.tsx`) |
| `src/api/` | typed fetch client (`client.ts`), React Query hooks (`hooks.ts`), response types (`types.ts`) |
| `src/state/` | zustand stores for the cleanup + ΔPDF view state |
| `src/colormaps/` | client-side colormap LUTs |

See [../docs/web.md](../docs/web.md) for the full reference — endpoints, the
binary slice envelope, and packaging.
