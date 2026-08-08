export default function MethodologyPage() {
  return (
    <main id="main-content" className="document-shell">
      <header className="document-header">
        <h1>A classification protocol fixed before training.</h1>
        <p>The task, split, model selection, calibration, quality gate, and limitations are recorded before a score exists.</p>
      </header>
      <div className="document-grid">
        <nav className="document-nav" aria-label="On this page">
          <a href="#task">Task boundary</a><a href="#data">Data protocol</a><a href="#model">Model</a><a href="#evaluation">Evaluation</a><a href="#release">Release gate</a>
        </nav>
        <article className="document-content">
          <section id="task">
            <h2>Task boundary</h2>
            <p>V1 classifies one Pakistani banknote denomination per image or crop across PKR 10, 20, 50, 100, 500, 1000, and 5000. It does not locate notes, count multiple notes, calculate totals, assess damage, or verify authenticity.</p>
          </section>
          <section id="data">
            <h2>Data protocol</h2>
            <p>UCP v1 supplies image-level training and internal evaluation. A label is accepted only when every source annotation in the image names one denomination; PKR 75 images are excluded. Exact and near duplicates remain in one deterministic 70/15/15 group split.</p>
            <p>Whole-note boxes from Abduls become an isolated crop set. Cross-source matches are removed, and this set is never used for training, calibration, or model selection.</p>
          </section>
          <section id="model">
            <h2>Model and ablation</h2>
            <p>Pretrained EfficientNet-B0 trains with seed 20260908. Nested levels use 10, 50, 100, and all available training images per class under matched augmented and non-augmented conditions. The infeasible 500 level remains recorded as omitted.</p>
            <p>The full-data condition is selected by validation macro F1, then validation calibration error within a 0.002 tie. Temperature scaling fits validation logits only.</p>
          </section>
          <section id="evaluation">
            <h2>Evaluation</h2>
            <p>Internal and cross-dataset reports include accuracy, balanced accuracy, macro and per-class F1, precision and recall, calibration error, Brier score, top-2 accuracy, confusion matrices, and fixed-test bootstrap intervals. The intervals exclude training-seed variance.</p>
          </section>
          <section id="release">
            <h2>Release gate</h2>
            <p>The API loads only a checksum-verified ONNX classifier whose evidence reaches 0.90 internal macro F1, 0.70 Abduls macro F1, and 0.75 recall for every internal class. Results remain publishable when a gate fails, but the console stays unavailable.</p>
            <p>A complete release contains the classifier, preprocessing and calibration metadata, canonical results, model card, and checksums. Missing, mismatched, or below-gate artifacts never produce simulated output.</p>
          </section>
        </article>
      </div>
    </main>
  );
}
