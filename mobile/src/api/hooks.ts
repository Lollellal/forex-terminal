import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CURRENT_USER_ID, get, post } from "./client";
import type {
  AllocationOverviewResponse,
  CloseReason,
  EmpireOverviewResponse,
  AccountBalanceProjectionResponse,
  JournalViewResponse,
  MarketSnapshotResponse,
  PortfolioResponse,
  WeeklyReportDownloadUrlResponse,
  WeeklyReportResponse,
} from "./types";

export function usePortfolio() {
  return useQuery({
    queryKey: ["portfolio", CURRENT_USER_ID],
    queryFn: () => get<PortfolioResponse>(`/users/${CURRENT_USER_ID}/portfolio`),
    refetchInterval: 60_000,
  });
}

export function useActiveTrades() {
  return useQuery({
    queryKey: ["allocations", CURRENT_USER_ID, "ACTIVE"],
    queryFn: () =>
      get<AllocationOverviewResponse[]>(
        `/allocations?user_id=${CURRENT_USER_ID}&status=ACTIVE`,
      ),
    refetchInterval: 60_000,
  });
}

export function useAllocationDetail(allocationId: string | undefined) {
  return useQuery({
    queryKey: ["allocation", allocationId],
    queryFn: () => get<AllocationOverviewResponse>(`/allocations/${allocationId}`),
    enabled: Boolean(allocationId),
  });
}

export function useAccountAllocations(accountId: string | undefined) {
  return useQuery({
    queryKey: ["allocations", "account", accountId],
    queryFn: () => get<AllocationOverviewResponse[]>(`/allocations?account_id=${accountId}`),
    enabled: Boolean(accountId),
  });
}

export function useMarketSnapshot() {
  return useQuery({
    queryKey: ["market-snapshot"],
    queryFn: () => get<MarketSnapshotResponse>(`/market-snapshot`),
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useJournal() {
  return useQuery({
    queryKey: ["journal", CURRENT_USER_ID],
    queryFn: () => get<JournalViewResponse[]>(`/journal?user_id=${CURRENT_USER_ID}`),
  });
}

export function useJournalEntry(allocationId: string | undefined) {
  return useQuery({
    queryKey: ["journal", "allocation", allocationId],
    queryFn: () => get<JournalViewResponse>(`/journal/${allocationId}`),
    enabled: Boolean(allocationId),
  });
}

export function useEmpireAccounts(empireId: string | undefined) {
  return useQuery({
    queryKey: ["empire-accounts", empireId],
    queryFn: () => get<AccountBalanceProjectionResponse[]>(`/empires/${empireId}/accounts`),
    enabled: Boolean(empireId),
  });
}

export function useAccount(accountId: string | undefined) {
  return useQuery({
    queryKey: ["account", accountId],
    queryFn: () => get<AccountBalanceProjectionResponse>(`/accounts/${accountId}/balance`),
    enabled: Boolean(accountId),
  });
}

export function useEmpireOverview(empireId: string | undefined) {
  return useQuery({
    queryKey: ["empire-overview", empireId],
    queryFn: () => get<EmpireOverviewResponse>(`/empires/${empireId}/overview`),
    enabled: Boolean(empireId),
  });
}

export function useWeeklyReports() {
  return useQuery({
    queryKey: ["weekly-reports", CURRENT_USER_ID],
    queryFn: () =>
      get<WeeklyReportResponse[]>(`/users/${CURRENT_USER_ID}/weekly-reports?limit=20`),
  });
}

export function useReportDownloadUrl(reportId: string | undefined) {
  return useQuery({
    queryKey: ["weekly-report-url", reportId],
    queryFn: () => get<WeeklyReportDownloadUrlResponse>(`/weekly-reports/${reportId}/download-url`),
    enabled: Boolean(reportId),
    staleTime: 5 * 60_000,
  });
}

export function useCloseAllocation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      allocationId,
      closeReason,
      realizedR,
    }: {
      allocationId: string;
      closeReason: CloseReason;
      realizedR: number;
    }) =>
      post(`/allocations/${allocationId}/close`, {
        close_reason: closeReason,
        realized_r: realizedR,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["journal"] });
      queryClient.invalidateQueries({ queryKey: ["allocations"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
    },
  });
}

export function useAddJournalNote() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ allocationId, text }: { allocationId: string; text: string }) =>
      post(`/journal-notes`, { related_allocation_id: allocationId, text, attachments: [] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["journal"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
    },
  });
}
