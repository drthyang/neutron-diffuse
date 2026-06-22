// Shared TanStack Query client.
//
// Exported from a module (not created inline in main.tsx) so non-React code —
// the pipeline store driving the in-browser Pyodide run — can invalidate viewer
// queries when fresh pipeline outputs land, without threading a hook through.

import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false },
  },
});
