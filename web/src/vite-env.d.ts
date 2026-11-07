/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of artifacts/models (a CDN in production). */
  readonly VITE_ARTIFACTS_URL?: string
  /** Folder name under artifacts/models to load by default. */
  readonly VITE_DEFAULT_MODEL?: string
}
