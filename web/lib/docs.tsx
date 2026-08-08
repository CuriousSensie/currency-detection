import fs from "node:fs";
import path from "node:path";
import type { ReactNode } from "react";

export type DocEntry = {
  slug: string;
  title: string;
  group: string;
  relativePath: string;
};

const repoRoot = findRepoRoot();
const docs = [
  ["README", "Project README", "Project", "README.md"],
  ["third-party-notices", "Third-party notices", "Project", "THIRD_PARTY_NOTICES.md"],
  ["api", "API", "Documentation", "docs/api.md"],
  ["architecture", "Architecture", "Documentation", "docs/architecture.md"],
  ["data-card", "Dataset card", "Documentation", "docs/data-card.md"],
  ["development", "Development", "Documentation", "docs/development.md"],
  ["limitations", "Limitations", "Documentation", "docs/limitations.md"],
  ["model-card", "Model card", "Documentation", "docs/model-card.md"],
  ["release", "Release", "Documentation", "docs/release.md"],
  ["research-protocol", "Research protocol", "Documentation", "docs/research-protocol.md"],
  ["training-and-evaluation", "Training and evaluation", "Documentation", "docs/training-and-evaluation.md"],
] satisfies ReadonlyArray<readonly [string, string, string, string]>;

const docEntries: DocEntry[] = docs.map(([slug, title, group, relativePath]) => ({ slug, title, group, relativePath }));

export function listDocs() {
  return docEntries.filter((doc) => fs.existsSync(path.join(/*turbopackIgnore: true*/ repoRoot, doc.relativePath)));
}

export function getDoc(slug: string) {
  const entry = listDocs().find((doc) => doc.slug === slug);
  if (!entry) return null;
  const absolutePath = path.join(/*turbopackIgnore: true*/ repoRoot, entry.relativePath);
  const raw = fs.readFileSync(absolutePath, "utf-8");
  const content = entry.relativePath.endsWith(".json") ? `\`\`\`json\n${raw.trim()}\n\`\`\`` : raw;
  return { ...entry, content };
}

export function groupedDocs() {
  return listDocs().reduce<Record<string, DocEntry[]>>((groups, doc) => {
    groups[doc.group] = [...(groups[doc.group] ?? []), doc];
    return groups;
  }, {});
}

export function MarkdownArticle({ markdown }: { markdown: string }) {
  return <>{parseBlocks(markdown).map((block, index) => renderBlock(block, index))}</>;
}

type Block =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "code"; language: string; text: string }
  | { type: "table"; rows: string[][] }
  | { type: "video"; src: string; title: string }
  | { type: "quote"; text: string };

function parseBlocks(markdown: string): Block[] {
  const lines = markdown.replaceAll("\r\n", "\n").split("\n");
  const blocks: Block[] = [];
  for (let index = 0; index < lines.length; ) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.startsWith("```")) {
      const language = line.slice(3).trim();
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        body.push(lines[index]);
        index += 1;
      }
      blocks.push({ type: "code", language, text: body.join("\n") });
      index += 1;
      continue;
    }
    const video = /^<video\s+(.+)><\/video>$/.exec(line.trim());
    if (video) {
      const src = /src="([^"]+)"/.exec(video[1])?.[1];
      const title = /title="([^"]+)"/.exec(video[1])?.[1] ?? "Demo video";
      if (src) blocks.push({ type: "video", src, title });
      index += 1;
      continue;
    }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      index += 1;
      continue;
    }
    if (line.startsWith(">")) {
      const body: string[] = [];
      while (index < lines.length && lines[index].startsWith(">")) {
        body.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push({ type: "quote", text: body.join(" ") });
      continue;
    }
    if (isTableLine(line)) {
      const rows: string[][] = [];
      while (index < lines.length && isTableLine(lines[index])) {
        const cells = lines[index].split("|").slice(1, -1).map((cell) => cell.trim());
        if (!cells.every((cell) => /^:?-{3,}:?$/.test(cell))) rows.push(cells);
        index += 1;
      }
      blocks.push({ type: "table", rows });
      continue;
    }
    if (/^(\s*[-*]\s+|\s*\d+\.\s+)/.test(line)) {
      const ordered = /^\s*\d+\.\s+/.test(line);
      const items: string[] = [];
      while (index < lines.length && /^(\s*[-*]\s+|\s*\d+\.\s+)/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*([-*]|\d+\.)\s+/, ""));
        index += 1;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    const body = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^(#{1,4})\s+/.test(lines[index])) {
      if (lines[index].startsWith("```") || isTableLine(lines[index]) || /^(\s*[-*]\s+|\s*\d+\.\s+)/.test(lines[index])) break;
      body.push(lines[index].trim());
      index += 1;
    }
    blocks.push({ type: "paragraph", text: body.join(" ") });
  }
  return blocks;
}

function renderBlock(block: Block, key: number): ReactNode {
  if (block.type === "heading") {
    const Tag = `h${Math.min(block.level + 1, 4)}` as "h2" | "h3" | "h4";
    return <Tag key={key}>{inline(block.text)}</Tag>;
  }
  if (block.type === "paragraph") return <p key={key}>{inline(block.text)}</p>;
  if (block.type === "quote") return <blockquote key={key}>{inline(block.text)}</blockquote>;
  if (block.type === "code") return <pre key={key} data-language={block.language}><code>{block.text}</code></pre>;
  if (block.type === "video") {
    return (
      <video key={key} controls muted playsInline preload="metadata" title={block.title}>
        <source src={normalizeHref(block.src)} type="video/mp4" />
      </video>
    );
  }
  if (block.type === "list") {
    const Tag = block.ordered ? "ol" : "ul";
    return <Tag key={key}>{block.items.map((item) => <li key={item}>{inline(item)}</li>)}</Tag>;
  }
  return (
    <table className="markdown-table" key={key}>
      <tbody>{block.rows.map((row) => <tr key={row.join("|")}>{row.map((cell) => <td key={cell}>{inline(cell)}</td>)}</tr>)}</tbody>
    </table>
  );
}

function inline(text: string) {
  const parts = text.split(/(`[^`]+`|!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\))/g);
  return parts.map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    const image = /^!\[([^\]]*)\]\(([^)]+)\)$/.exec(part);
    if (image) {
      return (
        // eslint-disable-next-line @next/next/no-img-element
        <img key={index} src={normalizeHref(image[2])} alt={image[1]} loading="lazy" />
      );
    }
    const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part);
    if (link) {
      const href = normalizeHref(link[2]);
      return <a key={index} href={href}>{link[1]}</a>;
    }
    return <span key={index}>{part}</span>;
  });
}

function isTableLine(line: string) {
  return line.trim().startsWith("|") && line.trim().endsWith("|");
}

function normalizeHref(href: string) {
  if (/^(https?:|mailto:|#|\/)/.test(href)) return href;
  if (href.startsWith("docs/assets/")) return `/${href}`;
  return `/docs`;
}

function findRepoRoot() {
  const candidates = [path.resolve(process.cwd(), ".."), process.cwd()];
  return candidates.find((candidate) => fs.existsSync(path.join(candidate, "README.md"))) ?? path.resolve(process.cwd(), "..");
}
