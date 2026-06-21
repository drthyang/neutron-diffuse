/// <reference types="vite/client" />

interface ImportMetaEnv {
  // "static" selects the backend-less GitHub Pages data layer (see staticData.ts);
  // unset/anything else uses the live FastAPI backend.
  readonly VITE_DATA_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
