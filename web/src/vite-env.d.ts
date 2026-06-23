/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Data-layer selector for the front-end:
  //   "pyodide" — run the real nebula3d pipeline in-browser via Pyodide, on the
  //               user's own file (the GitHub Pages build; see pyodideEngine.ts).
  //   unset/other — talk to the live FastAPI backend over /api.
  readonly VITE_DATA_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
