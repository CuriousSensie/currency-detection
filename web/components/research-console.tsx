"use client";

import { AlertTriangle, ArrowRight, Download, FileImage, LoaderCircle, ScanSearch, Upload } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { DragEvent, useCallback, useEffect, useRef, useState } from "react";

import { CameraCapture } from "./camera-capture";
import { classifyImage, fetchModelInfo } from "@/lib/api";
import type { ClassificationResponse, ModelInfo } from "@/lib/contracts";

const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

export function ResearchConsole() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchModelInfo(controller.signal)
      .then(setModel)
      .catch((reason: unknown) => {
        if (isAbortReason(reason)) return;
        const message = reason instanceof Error ? reason.message : "Unknown error";
        setError(`Model status unavailable. ${message}`);
      });
    return () => controller.abort();
  }, []);

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    },
    [],
  );

  const acceptFile = useCallback((candidate: File) => {
    if (!allowedTypes.has(candidate.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (candidate.size > 12 * 1024 * 1024) {
      setError("Choose an image smaller than 12 MB.");
      return;
    }
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const nextPreview = URL.createObjectURL(candidate);
    previewRef.current = nextPreview;
    setPreview(nextPreview);
    setFile(candidate);
    setResult(null);
    setError(null);
  }, []);

  useEffect(() => {
    function paste(event: ClipboardEvent) {
      const image = Array.from(event.clipboardData?.files ?? []).find((item) => allowedTypes.has(item.type));
      if (image) acceptFile(image);
    }
    window.addEventListener("paste", paste);
    return () => window.removeEventListener("paste", paste);
  }, [acceptFile]);

  async function classify() {
    if (!file) return setError("Choose an image before classifying it.");
    if (!model?.ready) return setError("No classifier has passed the registered quality gate yet.");
    setBusy(true);
    setError(null);
    try {
      setResult(await classifyImage(file));
      setResolving(true);
      window.setTimeout(() => setResolving(false), 700);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Classification failed. Check the service and retry.");
    } finally {
      setBusy(false);
    }
  }

  function downloadResult() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `pkr-classification-${result.request_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main id="main-content" className="console-shell">
      <section className="console-intro">
        <div>
          <h1>PKR note classifier</h1>
          <p>Upload or capture one Pakistani banknote. The service returns a calibrated denomination prediction, all class probabilities, latency, and model evidence.</p>
        </div>
        <div className={`readiness ${model?.ready ? "is-ready" : "is-waiting"}`}>
          <span aria-hidden="true" />
          <div><strong>{model?.ready ? "Model ready" : "Model unavailable"}</strong><small>{model?.ready ? model.model.classifier_version : "No simulated output"}</small></div>
        </div>
      </section>

      <section className="workbench classifier-workbench" aria-label="Banknote classification workspace">
        <div className="stage-column">
          <div
            className={`specimen-stage ${dragging ? "is-dragging" : ""} ${resolving ? "is-resolving" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event: DragEvent<HTMLDivElement>) => {
              event.preventDefault();
              setDragging(false);
              if (event.dataTransfer.files[0]) acceptFile(event.dataTransfer.files[0]);
            }}
          >
            {preview ? (
              <div className="image-plane">
                <Image src={preview} alt="Selected Pakistani banknote" fill unoptimized sizes="(max-width: 800px) 100vw, 70vw" />
                {busy && <div className="analysis-sweep"><LoaderCircle className="spin" size={20} /><span>Classifying with the verified model…</span></div>}
                {resolving && <div className="resolution-sweep" aria-hidden="true" />}
              </div>
            ) : (
              <button className="empty-stage" type="button" onClick={() => inputRef.current?.click()}>
                <span className="aperture"><ScanSearch size={42} /></span>
                <strong>Add one banknote image</strong>
                <span>Use one denomination per image. Drop, paste, choose, or capture.</span>
                <em><Upload size={16} /> Choose image</em>
              </button>
            )}
          </div>
          <div className="acquisition-bar">
            <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => event.target.files?.[0] && acceptFile(event.target.files[0])} />
            <button className="text-action" type="button" onClick={() => inputRef.current?.click()}><FileImage size={16} /> {file ? "Replace image" : "Choose image"}</button>
            <CameraCapture onCapture={acceptFile} />
            <span>{file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : "JPEG, PNG, WebP · 12 MB max"}</span>
          </div>
          {error && <p className="error-strip" role="alert"><AlertTriangle size={17} /><span>{error}</span></p>}
          {!model?.ready && model?.missing_artifacts[0] && (
            <div className="artifact-notice"><div><strong>Classifier is not available.</strong><p>{model.missing_artifacts[0]}</p></div><Link href="/docs/release">Release procedure <ArrowRight size={15} /></Link></div>
          )}
        </div>

        <aside className="instrument-rail" aria-label="Classification evidence">
          <div className="rail-section controls">
            <div className="section-heading"><h2>Input contract</h2><span>single denomination</span></div>
            <p className="rail-copy">Use one note or one note crop. Multi-note images are outside v1.</p>
            <button className="primary-action" type="button" disabled={!file || busy || !model?.ready} onClick={classify}>{busy ? <><LoaderCircle className="spin" size={17} /> Classifying</> : <><ScanSearch size={17} /> Classify note</>}</button>
          </div>

          <div className="rail-section classification-readout" aria-live="polite">
            <div className="section-heading"><h2>Classification</h2>{result && <button type="button" onClick={downloadResult} aria-label="Download JSON result"><Download size={16} /></button>}</div>
            {result ? (
              <div className="primary-verdict">
                <span>Most probable denomination</span>
                <strong>PKR {result.prediction.denomination_pkr.toLocaleString()}</strong>
                <em>{(result.prediction.confidence * 100).toFixed(1)}% calibrated confidence</em>
              </div>
            ) : <p className="rail-empty">Results appear after a ready model classifies an image.</p>}
          </div>

          <div className="rail-section probability-ledger">
            <div className="section-heading"><h2>Probability distribution</h2><span>all seven classes</span></div>
            <div className="probability-list">
              {(result?.prediction.probabilities ?? model?.denominations.map((denomination) => ({ denomination_pkr: denomination, probability: 0 })) ?? []).map((item) => (
                <div className={item.denomination_pkr === result?.prediction.denomination_pkr ? "is-leading" : ""} key={item.denomination_pkr}>
                  <span>PKR {item.denomination_pkr.toLocaleString()}</span>
                  <i><b style={{ transform: `scaleX(${item.probability})` }} /></i>
                  <output>{result ? `${(item.probability * 100).toFixed(1)}%` : "—"}</output>
                </div>
              ))}
            </div>
          </div>

          <div className="rail-section provenance">
            <div className="section-heading"><h2>Run evidence</h2><span>{result ? result.request_id.slice(0, 8) : "no run"}</span></div>
            <dl>
              <div><dt>Classifier</dt><dd>{result?.model.classifier_version ?? model?.model.classifier_version ?? "—"}</dd></div>
              <div><dt>Runtime</dt><dd>{result?.model.runtime ?? model?.model.runtime ?? "—"}</dd></div>
              <div><dt>Latency</dt><dd>{result ? `${result.timings_ms.total.toFixed(1)} ms` : "—"}</dd></div>
              <div><dt>Artifact</dt><dd>{shortHash(Object.values(model?.artifact_checksums ?? {})[0])}</dd></div>
            </dl>
          </div>
        </aside>
      </section>

      <section className="truth-strip">
        <div><strong>Scope</strong><p>Seven-class denomination prediction for one note image.</p></div>
        <div><strong>Limits</strong><p>No multi-note detection, condition grading, or authenticity check.</p></div>
        <Link href="/docs">Open documentation <ArrowRight size={16} /></Link>
      </section>
    </main>
  );
}

function shortHash(value?: string) {
  return value ? `${value.slice(0, 10)}…` : "—";
}

function isAbortReason(reason: unknown) {
  return reason instanceof DOMException && reason.name === "AbortError";
}
