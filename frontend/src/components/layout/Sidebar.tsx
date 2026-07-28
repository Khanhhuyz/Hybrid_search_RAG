"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  MessageSquare,
  Share2,
  Search,
  Cpu,
} from "lucide-react";

const navItems = [
  { href: "/",        icon: LayoutDashboard, label: "Dashboard" },
  { href: "/chat",    icon: MessageSquare,   label: "Chat" },
  { href: "/graph",   icon: Share2,          label: "Graph" },
  { href: "/search",  icon: Search,          label: "Search" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-16 lg:w-60 flex-shrink-0 flex flex-col h-screen bg-[#12141a] border-r border-[#252836]">
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-[#252836] gap-3 flex-shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center flex-shrink-0">
          <Cpu size={16} className="text-white" />
        </div>
        <span className="hidden lg:block font-bold text-sm gradient-text tracking-wide">GRAG System</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-1 px-2">
        {navItems.map(({ href, icon: Icon, label }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 group ${
                active
                  ? "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30"
                  : "text-zinc-400 hover:text-zinc-100 hover:bg-white/5"
              }`}
            >
              <Icon size={18} className={active ? "text-indigo-400" : "text-zinc-500 group-hover:text-zinc-300"} />
              <span className="hidden lg:block">{label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-[#252836]">
        <div className="hidden lg:flex items-center gap-2 text-xs text-zinc-600">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>Local Model Active</span>
        </div>
      </div>
    </aside>
  );
}
