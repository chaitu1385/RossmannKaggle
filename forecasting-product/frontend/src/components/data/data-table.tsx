"use client";

import { useMemo, useState } from "react";
import { ArrowUpDown, ChevronLeft, ChevronRight } from "lucide-react";

interface ColumnDef {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (value: unknown) => React.ReactNode;
}

interface DataTableProps {
  columns: ColumnDef[];
  data: Record<string, unknown>[];
  pageSize?: number;
}

export function DataTable({ columns, data, pageSize = 10 }: DataTableProps) {
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number")
        return sortAsc ? av - bv : bv - av;
      const sa = String(av);
      const sb = String(bv);
      return sortAsc ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
  }, [data, sortKey, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const rows = sorted.slice(safePage * pageSize, (safePage + 1) * pageSize);

  function handleSort(key: string) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
    setPage(0);
  }

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-3 py-2 text-left text-xs font-medium text-muted-foreground ${
                    col.sortable ? "cursor-pointer select-none hover:text-foreground" : ""
                  }`}
                  onClick={col.sortable ? () => handleSort(col.key) : undefined}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {col.sortable && (
                      <ArrowUpDown className={`h-3 w-3 ${sortKey === col.key ? "text-foreground" : "text-muted-foreground/40"}`} />
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-3 py-6 text-center text-muted-foreground">
                  No data
                </td>
              </tr>
            ) : (
              rows.map((row, i) => (
                <tr key={i} className="border-b last:border-0 hover:bg-muted/30">
                  {columns.map((col) => (
                    <td key={col.key} className="px-3 py-2">
                      {col.render ? col.render(row[col.key]) : String(row[col.key] ?? "")}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>
            {safePage * pageSize + 1}–{Math.min((safePage + 1) * pageSize, sorted.length)} of{" "}
            {sorted.length}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage(Math.max(0, safePage - 1))}
              disabled={safePage === 0}
              className="rounded p-1 hover:bg-muted disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span>
              {safePage + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, safePage + 1))}
              disabled={safePage >= totalPages - 1}
              className="rounded p-1 hover:bg-muted disabled:opacity-30"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
