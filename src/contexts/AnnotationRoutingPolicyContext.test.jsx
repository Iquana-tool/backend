import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  AnnotationRoutingPolicyProvider,
  useAnnotationRoutingPolicy,
} from "./AnnotationRoutingPolicyContext";
import { getInferenceRoutingPolicy } from "../api/inference";

vi.mock("../api/inference", () => ({
  getInferenceRoutingPolicy: vi.fn(),
}));

function PolicyProbe({ datasetId, name }) {
  const state = useAnnotationRoutingPolicy(datasetId);
  return (
    <div data-testid={name}>
      {state.policyLoading ? "loading" : state.policyError || state.policy?.dataset_id || "none"}
    </div>
  );
}

function CurrentDatasetPolicyHarness({ currentDataset, children }) {
  return (
    <AnnotationRoutingPolicyProvider datasetId={currentDataset?.id}>
      {children}
    </AnnotationRoutingPolicyProvider>
  );
}

describe("AnnotationRoutingPolicyProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads once and shares the same policy across all workspace consumers", async () => {
    getInferenceRoutingPolicy.mockResolvedValue({ dataset_id: 42, bindings: [] });

    render(
      <AnnotationRoutingPolicyProvider datasetId={42}>
        <PolicyProbe datasetId={42} name="first" />
        <PolicyProbe datasetId={42} name="second" />
        <PolicyProbe datasetId={42} name="third" />
      </AnnotationRoutingPolicyProvider>
    );

    await waitFor(() => expect(screen.getByTestId("first")).toHaveTextContent("42"));
    expect(screen.getByTestId("second")).toHaveTextContent("42");
    expect(screen.getByTestId("third")).toHaveTextContent("42");
    expect(getInferenceRoutingPolicy).toHaveBeenCalledTimes(1);
    expect(getInferenceRoutingPolicy).toHaveBeenCalledWith(42);
  });

  it("invalidates the previous policy immediately when the dataset changes", async () => {
    let resolveSecond;
    getInferenceRoutingPolicy
      .mockResolvedValueOnce({ dataset_id: 42, bindings: [] })
      .mockImplementationOnce(
        () => new Promise((resolve) => {
          resolveSecond = resolve;
        })
      );

    const { rerender } = render(
      <AnnotationRoutingPolicyProvider datasetId={42}>
        <PolicyProbe datasetId={42} name="probe" />
      </AnnotationRoutingPolicyProvider>
    );
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("42"));

    rerender(
      <AnnotationRoutingPolicyProvider datasetId={43}>
        <PolicyProbe datasetId={43} name="probe" />
      </AnnotationRoutingPolicyProvider>
    );
    expect(screen.getByTestId("probe")).toHaveTextContent("loading");

    resolveSecond({ dataset_id: 43, bindings: [] });
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("43"));
  });

  it("resolves a current dataset when the route has no dataset param with one shared request", async () => {
    getInferenceRoutingPolicy.mockResolvedValue({ dataset_id: 42, bindings: [] });

    const { rerender } = render(
      <CurrentDatasetPolicyHarness currentDataset={null}>
        <PolicyProbe datasetId={42} name="first" />
        <PolicyProbe datasetId={42} name="second" />
      </CurrentDatasetPolicyHarness>
    );

    expect(screen.getByTestId("first")).toHaveTextContent("loading");

    rerender(
      <CurrentDatasetPolicyHarness currentDataset={{ id: 42 }}>
        <PolicyProbe datasetId={42} name="first" />
        <PolicyProbe datasetId={42} name="second" />
      </CurrentDatasetPolicyHarness>
    );

    await waitFor(() => expect(screen.getByTestId("first")).toHaveTextContent("42"));
    expect(screen.getByTestId("second")).toHaveTextContent("42");
    expect(getInferenceRoutingPolicy).toHaveBeenCalledTimes(1);
    expect(getInferenceRoutingPolicy).toHaveBeenCalledWith(42);
  });
});
