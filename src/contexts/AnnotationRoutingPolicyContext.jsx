import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { getInferenceRoutingPolicy } from "../api/inference";

const AnnotationRoutingPolicyContext = createContext(null);

const sameDataset = (left, right) =>
  left != null && right != null && String(left) === String(right);

const initialState = (datasetId) => ({
  datasetId,
  status: datasetId != null ? "loading" : "idle",
  policy: null,
  error: null,
});

const getPolicyErrorMessage = (error) =>
  error?.message || "Failed to load model routing policy";

function usePolicyLoader(datasetId, enabled) {
  const [state, setState] = useState(() => initialState(datasetId));

  useEffect(() => {
    if (!enabled) return undefined;

    let cancelled = false;
    setState(initialState(datasetId));

    if (datasetId == null) {
      return () => {
        cancelled = true;
      };
    }

    getInferenceRoutingPolicy(datasetId)
      .then((policy) => {
        if (!cancelled) {
          setState({ datasetId, status: "loaded", policy: policy || null, error: null });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setState({
            datasetId,
            status: "error",
            policy: null,
            error: getPolicyErrorMessage(error),
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId, enabled]);

  return useMemo(() => ({ requestedDatasetId: datasetId, ...state }), [datasetId, state]);
}

const scopeStateToDataset = (source, datasetId) => {
  const matches = sameDataset(source?.datasetId, datasetId);
  const status = matches ? source.status : datasetId != null ? "loading" : "idle";
  const policyReady = status === "loaded";
  const policyError = status === "error" ? source.error : null;

  return {
    policy: policyReady ? source.policy : null,
    policyReady,
    policyResolved: status === "loaded" || status === "error" || status === "idle",
    policyLoading: status === "loading",
    policyError,
  };
};

/**
 * Loads one dataset routing policy for the complete annotation workspace.
 * Consumers still support an isolated fallback loader so hook unit tests and
 * reusable components remain self-contained outside the page provider.
 */
export function AnnotationRoutingPolicyProvider({ datasetId, children }) {
  const state = usePolicyLoader(datasetId, true);
  return (
    <AnnotationRoutingPolicyContext.Provider value={state}>
      {children}
    </AnnotationRoutingPolicyContext.Provider>
  );
}

export function useAnnotationRoutingPolicy(datasetId) {
  const sharedState = useContext(AnnotationRoutingPolicyContext);
  const localState = usePolicyLoader(datasetId, sharedState == null);
  return scopeStateToDataset(sharedState || localState, datasetId);
}
