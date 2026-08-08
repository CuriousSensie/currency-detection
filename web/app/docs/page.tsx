import Link from "next/link";

import { groupedDocs } from "@/lib/docs";

export default function DocsIndexPage() {
  const groups = groupedDocs();
  return (
    <main id="main-content" className="document-shell docs-shell">
      <header className="document-header">
        <h1>Documentation</h1>
        <p>Project notes, protocol, model evidence, API details, and decision records from the repository.</p>
      </header>
      <div className="docs-index">
        {Object.entries(groups).map(([group, items]) => (
          <section key={group}>
            <h2>{group}</h2>
            <ul>
              {items.map((item) => (
                <li key={item.slug}>
                  <Link href={`/docs/${item.slug}`}>
                    <span>{item.title}</span>
                    <small>{item.relativePath}</small>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </main>
  );
}
