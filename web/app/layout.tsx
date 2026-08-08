import "@fontsource/ibm-plex-mono/400.css";
import "./globals.css";

import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "PKR Vision",
  description: "Classify one Pakistani banknote and inspect calibrated model evidence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body>
        <a className="skip-link" href="#main-content">Skip to content</a>
        <header className="masthead">
          <Link className="wordmark" href="/" aria-label="PKR Vision home">
            <span className="wordmark-mark" aria-hidden="true"><i /><i /><i /></span>
            <span>PKR Vision</span>
          </Link>
          <nav aria-label="Primary navigation">
            <Link href="/">Console</Link>
            <Link href="/evidence">Evidence</Link>
            <Link href="/methodology">Methodology</Link>
            <Link href="/docs">Docs</Link>
          </nav>
          <span className="masthead-note">Single note · seven classes</span>
        </header>
        {children}
        <footer className="site-footer">
          <p>Research software by Hasnaat Khalid.</p>
          <p>Single-note denomination classification. No detection or authenticity claims.</p>
        </footer>
      </body>
    </html>
  );
}
