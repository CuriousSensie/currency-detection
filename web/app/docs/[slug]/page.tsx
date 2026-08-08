import Link from "next/link";
import { notFound } from "next/navigation";

import { getDoc, listDocs, MarkdownArticle } from "@/lib/docs";

export const dynamicParams = false;

export function generateStaticParams() {
  return listDocs().map((doc) => ({ slug: doc.slug }));
}

export default async function DocPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = getDoc(slug);
  if (!doc) notFound();
  return (
    <main id="main-content" className="document-shell markdown-shell">
      <header className="document-header">
        <h1>{doc.title}</h1>
        <p>{doc.relativePath}</p>
      </header>
      <div className="document-grid">
        <nav className="document-nav" aria-label="Documentation navigation">
          <Link href="/docs">All docs</Link>
          <Link href="/evidence">Evidence</Link>
          <Link href="/methodology">Methodology</Link>
        </nav>
        <article className="document-content markdown-content">
          <MarkdownArticle markdown={doc.content} />
        </article>
      </div>
    </main>
  );
}
