// Shared dataset selection for every page in the console.

import { useEffect } from "react";
import { create } from "zustand";

import type { Dataset } from "../api/types";

interface DatasetState {
  datasetId?: string;
  setDataset: (id: string) => void;
  resetDataset: () => void;
}

export const useDatasetStore = create<DatasetState>((set) => ({
  datasetId: undefined,
  setDataset: (datasetId) => set({ datasetId }),
  resetDataset: () => set({ datasetId: undefined }),
}));

export function useInitializeDataset(datasets: readonly Dataset[]) {
  const datasetId = useDatasetStore((s) => s.datasetId);
  const setDataset = useDatasetStore((s) => s.setDataset);

  useEffect(() => {
    if (!datasetId && datasets.length) setDataset(datasets[0].id);
  }, [datasetId, datasets, setDataset]);
}
