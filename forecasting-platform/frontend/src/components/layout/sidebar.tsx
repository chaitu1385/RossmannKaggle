"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Upload,
  Search,
  ArrowLeftRight,
  GitBranch,
  Trophy,
  TrendingUp,
  Activity,
  Presentation,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "@/lib/constants";

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  Upload,
  Search,
  ArrowLeftRight,
  GitBranch,
  Trophy,
  TrendingUp,
  Activity,
  Presentation,
};

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "flex flex-col border-r bg-sidebar text-sidebar-foreground transition-all duration-200",
        collapsed ? "w-16" : "w-64",
      )}
    >
      {/* Logo area */}
      <div className="flex h-14 items-center border-b px-4">
        {!collapsed && (
          <span className="text-sm font-bold tracking-tight">
            Forecasting Platform
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto rounded-md p-1 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-2">
        {NAV_ITEMS.map((item, idx) => {
          const Icon = ICONS[item.icon];
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "hover:bg-sidebar-accent/10 text-sidebar-foreground/70 hover:text-sidebar-foreground",
              )}
              title={collapsed ? item.title : undefined}
            >
              {Icon && <Icon className="h-4 w-4 shrink-0" />}
              {!collapsed && (
                <span className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground w-4">
                    {idx + 1}
                  </span>
                  {item.title}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="border-t p-4">
          <p className="text-xs text-muted-foreground">
            Multi-frequency S&amp;OP forecasting
          </p>
        </div>
      )}
    </aside>
  );
}
