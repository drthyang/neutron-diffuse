// TanStack Query hooks over the API client.

import { keepPreviousData, useQuery } from "@tanstack/react-query";

import {
  fetchDatasets,
  fetchDpdfMeta,
  fetchDpdfSlice,
  fetchMeta,
  fetchSlice,
} from "./client";

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

export function useSlice(
  volumeId: string | undefined,
  plane: string,
  value: number,
  interp = false,
) {
  return useQuery({
    queryKey: ["slice", volumeId, plane, value, interp],
    queryFn: () => fetchSlice(volumeId as string, plane, value, interp),
    enabled: Boolean(volumeId),
    placeholderData: keepPreviousData,
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
