// TanStack Query hooks over the API client.

import { keepPreviousData, useQuery } from "@tanstack/react-query";

import {
  fetchDatasets,
  fetchDpdfMeta,
  fetchDpdfSlice,
  fetchHealth,
  fetchMeta,
} from "./client";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 15_000,
    retry: false,
  });
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: fetchDatasets });
}

export function useMeta(volumeId: string | undefined) {
  return useQuery({
    queryKey: ["meta", volumeId],
    queryFn: () => fetchMeta(volumeId as string),
    enabled: Boolean(volumeId),
  });
}

export function useDpdfMeta(volumeId: string | undefined) {
  return useQuery({
    queryKey: ["dpdfMeta", volumeId],
    queryFn: () => fetchDpdfMeta(volumeId as string),
    enabled: Boolean(volumeId),
  });
}

export function useDpdfSlice(
  volumeId: string | undefined,
  plane: string,
  value: number,
) {
  return useQuery({
    queryKey: ["dpdfSlice", volumeId, plane, value],
    queryFn: () => fetchDpdfSlice(volumeId as string, plane, value),
    enabled: Boolean(volumeId),
    placeholderData: keepPreviousData,
  });
}
