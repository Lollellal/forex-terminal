import { NavLink } from "react-router-dom";
import { Home, LineChart, NotebookPen, Layers, Newspaper } from "lucide-react";

const TABS = [
  { to: "/", label: "Home", icon: Home, end: true },
  { to: "/trades", label: "Trades", icon: LineChart, end: false },
  { to: "/journal", label: "Journal", icon: NotebookPen, end: false },
  { to: "/empire", label: "Empire", icon: Layers, end: false },
  { to: "/reports", label: "Reports", icon: Newspaper, end: false },
];

export function BottomNav() {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-20 border-t border-hairline bg-card/95 backdrop-blur">
      <div className="mx-auto flex max-w-md items-stretch justify-between px-2 pb-[env(safe-area-inset-bottom)]">
        {TABS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex flex-1 flex-col items-center gap-1 py-2.5 text-[11px] font-medium transition-colors ${
                isActive ? "text-primary" : "text-ink-soft"
              }`
            }
          >
            <Icon size={22} strokeWidth={1.75} />
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
