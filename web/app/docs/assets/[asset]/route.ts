import fs from "node:fs";
import path from "node:path";

import { notFound } from "next/navigation";
import type { NextRequest } from "next/server";

export const dynamicParams = false;

const allowedAssets = new Set(["demo.webp", "demo.mp4"]);

export function generateStaticParams() {
  return [...allowedAssets].map((asset) => ({ asset }));
}

export async function GET(_request: NextRequest, { params }: { params: Promise<{ asset: string }> }) {
  const { asset } = await params;
  if (!allowedAssets.has(asset)) notFound();
  const repoRoot = findRepoRoot();
  const assetPath = path.join(/*turbopackIgnore: true*/ repoRoot, "docs", "assets", asset);
  if (!fs.existsSync(assetPath)) notFound();
  const contentType = asset.endsWith(".mp4") ? "video/mp4" : "image/webp";
  return new Response(fs.readFileSync(assetPath), {
    headers: {
      "content-type": contentType,
      "cache-control": "public, max-age=31536000, immutable",
    },
  });
}

function findRepoRoot() {
  const candidates = [path.resolve(process.cwd(), ".."), process.cwd()];
  return candidates.find((candidate) => fs.existsSync(path.join(candidate, "README.md"))) ?? path.resolve(process.cwd(), "..");
}
