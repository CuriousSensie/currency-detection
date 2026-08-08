import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ResearchConsole } from "./research-console";
import { fetchModelInfo } from "@/lib/api";

const modelInfo = {
  ready: false,
  model: {
    classifier_version: "not-installed",
    runtime: "onnx-cpu",
  },
  denominations: [10, 20, 50, 100, 500, 1000, 5000],
  preprocessing: { size: [224, 224] },
  artifact_checksums: {},
  quality_gate: { passed: false },
  training_data_summary: { status: "not_run" },
  evaluation_summary: { status: "not_run" },
  limitations: [],
  missing_artifacts: ["Model manifest is missing."],
};

vi.mock("@/lib/api", () => ({
  fetchModelInfo: vi.fn(),
  classifyImage: vi.fn(),
}));

describe("ResearchConsole", () => {
  beforeEach(() => {
    vi.mocked(fetchModelInfo).mockReset();
    vi.mocked(fetchModelInfo).mockResolvedValue(modelInfo);
  });

  it("does not invent output when model artifacts are absent", async () => {
    render(<ResearchConsole />);
    expect(screen.getByRole("heading", { name: /PKR note classifier/i })).toBeInTheDocument();
    expect(await screen.findByText("Model unavailable")).toBeInTheDocument();
    expect(screen.getByText("No simulated output")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Classify note" })).toBeDisabled();
  });

  it("ignores aborted model status requests", async () => {
    vi.mocked(fetchModelInfo).mockRejectedValue(new DOMException("signal is aborted without reason", "AbortError"));
    render(<ResearchConsole />);
    await waitFor(() => expect(fetchModelInfo).toHaveBeenCalled());
    expect(screen.queryByText(/Model status unavailable/i)).not.toBeInTheDocument();
  });
});
