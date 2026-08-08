import benchmark from "@/../artifacts/results/cpu-benchmark.json";
import ablation from "@/../artifacts/results/classification-ablation.json";
import external from "@/../artifacts/results/external-validation.json";

const records = [ablation, external, benchmark];
const metricLabels: Record<string, string> = {
  selected_test_macro_f1: "UCP macro F1",
  selected_test_accuracy: "UCP accuracy",
  macro_f1: "Macro F1",
  accuracy: "Accuracy",
  warm_combined_mean_ms: "Mean latency",
  warm_combined_p95_ms: "P95 latency",
  sequential_throughput_images_per_second: "Throughput",
};

function formatMetric(key: string, value: number) {
  const label = metricLabels[key] ?? key.replaceAll("_", " ");
  if (key.endsWith("_ms")) return `${label}: ${value.toFixed(2)} ms`;
  if (key === "sequential_throughput_images_per_second") return `${label}: ${value.toFixed(1)} img/s`;
  if (key === "model_size_bytes") return `${label}: ${(value / 1024 / 1024).toFixed(1)} MB`;
  if (key === "process_peak_rss_kib") return `${label}: ${(value / 1024).toFixed(1)} MB RSS`;
  return `${label}: ${value.toFixed(4)}`;
}

export default function EvidencePage() {
  return (
    <main id="main-content" className="document-shell">
      <header className="document-header"><h1>Evidence with a checksum trail.</h1><p>This page reads canonical experiment records. Completed rows must carry configuration, dataset, checkpoint, and metric evidence before they appear here.</p></header>
      <div className="document-grid">
        <nav className="document-nav" aria-label="On this page"><a href="#ledger">Experiment ledger</a><a href="#interpretation">Interpretation</a></nav>
        <article className="document-content">
          <section id="ledger"><h2>Experiment ledger</h2><table className="evidence-table"><thead><tr><th>Experiment</th><th>Kind</th><th>Status</th><th>Metrics</th></tr></thead><tbody>{records.map((record) => <tr key={record.experiment_id}><td>{record.experiment_id}</td><td>{record.kind.replaceAll("_", " ")}</td><td><span className={`status-${record.status.replace("_", "-")}`}>{record.status.replace("_", " ")}</span></td><td>{Object.entries(record.metrics).length ? Object.entries(record.metrics).slice(0, 3).map(([key, value]) => <span className="metric-chip" key={key}>{formatMetric(key, value)}</span>) : "—"}</td></tr>)}</tbody></table></section>
          <section id="interpretation"><h2>Interpretation</h2><p>The UCP and Abduls archives were audited, all eight feasible one-seed ablation runs completed, and the selected full-data augmentation run passed the preregistered quality gate. ONNX agreement is complete on the fixed UCP test set.</p><p>Abduls is isolated for cross-dataset evaluation. It is public benchmark data and must not be presented as user-collected real-world validation.</p></section>
        </article>
      </div>
    </main>
  );
}
