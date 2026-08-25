import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prism",
  description: "Claude-native LLM gateway — traces, caching, evaluation, cost.",
};

const NAV = [
  ["/", "Overview"],
  ["/traces", "Traces"],
  ["/cache", "Caching"],
  ["/evals", "Evaluation"],
  ["/prompts", "Prompts"],
] as const;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <nav className="sidebar">
            <div className="brand">Prism</div>
            <div className="brand-sub">LLM gateway</div>
            <div className="nav">
              {NAV.map(([href, label]) => (
                <Link key={href} href={href}>
                  {label}
                </Link>
              ))}
            </div>
          </nav>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
