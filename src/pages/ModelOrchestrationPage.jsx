import React, { useState, useEffect, useMemo, useRef } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Cpu, Loader2, ArrowLeft, AlertTriangle } from "lucide-react";
import DatasetManagementLayout from "../components/datasets/gallery/DatasetManagementLayout";
import { useDataset } from "../contexts/DatasetContext";
import {
    getInferenceModelCatalog,
    getInferenceRoutingPolicy,
    updateInferenceRoutingPolicy,
    deleteInferenceRoutingPolicy,
} from "../api/inference";
import { fetchLabels } from "../api/labels";
import ModelOrchestrationPanel from "../components/inference/ModelOrchestrationPanel";
import { calculateCoverage } from "../components/inference/orchestration/orchestrationViewModel";
import { usePermissions } from "../hooks/usePermissions";
import { Permission } from "../utils/permissions";

/**
 * Model Orchestration Page
 *
 * Dedicated dataset management page for configuring dataset-level task and label
 * routing policies. These policies serve as canonical defaults for interactive
 * annotation tools, cross-image suggestions, and batch inference runs.
 */
export default function ModelOrchestrationPage() {
    const { datasetId } = useParams();
    const { currentDataset, loading: datasetLoading } = useDataset();
    const navigate = useNavigate();
    const { can } = usePermissions(datasetId);

    const canEdit = Boolean(can(Permission.AI_BATCH_INFER));
    const canView = Boolean(can(Permission.AI_INTERACTIVE) || can(Permission.AI_BATCH_INFER));

    const [labelsById, setLabelsById] = useState({});
    const [catalog, setCatalog] = useState({ models: [], retrieval_strategies: [] });
    const [policy, setPolicy] = useState(null);
    const [isSaving, setIsSaving] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    const datasetIdRef = useRef(datasetId);
    datasetIdRef.current = datasetId;
    const isMountedRef = useRef(true);

    useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
        };
    }, []);

    useEffect(() => {
        if (!datasetId || !canView) {
            if (!datasetLoading) {
                setIsLoading(false);
            }
            return undefined;
        }
        let cancelled = false;
        setIsLoading(true);
        setError(null);
        setIsSaving(false);
        setIsDeleting(false);
        setPolicy(null);
        setLabelsById({});
        setCatalog({ models: [], retrieval_strategies: [] });

        (async () => {
            try {
                const [labelResponse, catalogResponse, policyResponse] = await Promise.all([
                    fetchLabels(datasetId),
                    getInferenceModelCatalog(datasetId),
                    getInferenceRoutingPolicy(datasetId),
                ]);
                if (cancelled || !isMountedRef.current || datasetIdRef.current !== datasetId) return;

                const loadedLabels = labelResponse?.labels?.id_to_label_object || {};
                const loadedCatalog = catalogResponse || { models: [], retrieval_strategies: [] };

                setLabelsById(loadedLabels);
                setCatalog(loadedCatalog);
                setPolicy(policyResponse);
            } catch (e) {
                if (!cancelled && isMountedRef.current && datasetIdRef.current === datasetId) {
                    setError(e.message || "Could not load the models or routing policy for this dataset.");
                }
            } finally {
                if (!cancelled && isMountedRef.current && datasetIdRef.current === datasetId) {
                    setIsLoading(false);
                }
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [datasetId, canView, datasetLoading]);

    const handleSavePolicy = async (bindings) => {
        const targetDatasetId = datasetId;
        setError(null);
        setIsSaving(true);
        try {
            const updated = await updateInferenceRoutingPolicy({
                dataset_id: Number(targetDatasetId),
                bindings,
            });
            if (!isMountedRef.current || datasetIdRef.current !== targetDatasetId) return;
            setPolicy(updated);
            return updated;
        } catch (e) {
            if (isMountedRef.current && datasetIdRef.current === targetDatasetId) {
                setError(e.message || "Failed to save dataset routing policy.");
            }
            throw e;
        } finally {
            if (isMountedRef.current && datasetIdRef.current === targetDatasetId) {
                setIsSaving(false);
            }
        }
    };

    const handleDeletePolicy = async () => {
        const targetDatasetId = datasetId;
        setError(null);
        setIsDeleting(true);
        try {
            await deleteInferenceRoutingPolicy(targetDatasetId);
            if (!isMountedRef.current || datasetIdRef.current !== targetDatasetId) return;
            setPolicy(null);
        } catch (e) {
            if (isMountedRef.current && datasetIdRef.current === targetDatasetId) {
                setError(e.message || "Failed to clear dataset routing policy.");
            }
            throw e;
        } finally {
            if (isMountedRef.current && datasetIdRef.current === targetDatasetId) {
                setIsDeleting(false);
            }
        }
    };

    const activeCoverage = useMemo(
        () => calculateCoverage(policy?.bindings || [], labelsById, catalog).overall,
        [policy?.bindings, labelsById, catalog]
    );

    const formatSavedDate = (dateStr) => {
        if (!dateStr) return null;
        try {
            const d = new Date(dateStr);
            return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) + ", " +
                   d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        } catch {
            return dateStr;
        }
    };

    return (
        <DatasetManagementLayout>
            <div className="flex flex-col h-full bg-app min-h-0 overflow-hidden">
                {/* Header */}
                <header className="py-4 px-6 border-b border-ln bg-p1 flex flex-wrap items-center justify-between gap-4 flex-shrink-0">
                    <div className="flex items-center gap-3 min-w-0">
                        <div className="p-2.5 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-400 shrink-0">
                            <Cpu size={22} />
                        </div>

                        <div className="min-w-0">
                            {/* Breadcrumb */}
                            <div className="flex items-center gap-1.5 text-xs text-t3 mb-0.5">
                                <span className="text-t2 font-medium truncate">
                                    {currentDataset?.name || `Dataset #${datasetId}`}
                                </span>
                                <span>&gt;</span>
                                <Link
                                    to={`/dataset/${datasetId}/datamanagement`}
                                    className="hover:text-t1 transition hover:underline"
                                >
                                    Dataset Management
                                </Link>
                            </div>
                            <h1 className="text-xl font-bold text-t1 leading-tight">
                                Model Orchestration
                            </h1>
                            <p className="text-xs text-t3 mt-0.5 line-clamp-1">
                                One cell per task and label. Configured routes take precedence; unconfigured cells use the task default when available.
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-5 shrink-0">
                        {/* Configured routes stats block */}
                        <div className="text-right hidden sm:block">
                            <div className="text-[10px] font-bold uppercase tracking-wider text-t3">
                                CONFIGURED ROUTES
                            </div>
                            <div className="text-sm font-bold text-t1 font-mono">
                                {datasetLoading || isLoading
                                    ? "Loading…"
                                    : `${activeCoverage.bound} / ${activeCoverage.possible}`}
                            </div>
                        </div>

                        <button
                            type="button"
                            onClick={() => navigate(`/dataset/${datasetId}/datamanagement`)}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-ln rounded-lg bg-well/40 text-t2 hover:text-t1 hover:bg-well transition shadow-2xs"
                        >
                            <span>Dataset Management</span>
                        </button>
                    </div>
                </header>

                {/* Main Content Area */}
                <main className="flex-1 overflow-y-auto p-6 pb-24">
                    {error && (
                        <div className="mb-4 p-3.5 bg-errBg border border-errLn rounded-xl text-xs text-err flex items-center gap-2 w-full">
                            <AlertTriangle size={15} className="shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}

                    {datasetLoading || (isLoading && canView) ? (
                        <div className="flex items-center justify-center p-12 text-sm text-t3 gap-2">
                            <Loader2 className="w-5 h-5 animate-spin text-ac" />
                            <span>Loading dataset models, routing policy, and labels…</span>
                        </div>
                    ) : !canView ? (
                        <div className="max-w-4xl mx-auto p-6 bg-p1 border border-ln rounded-3xl text-center space-y-3">
                            <h2 className="text-sm font-semibold text-t1">Access Restricted</h2>
                            <p className="text-xs text-t3">
                                You do not have permission to view or configure model orchestration for this dataset.
                            </p>
                            <button
                                type="button"
                                onClick={() => navigate(`/dataset/${datasetId}/datamanagement`)}
                                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium bg-accent text-white rounded-xl hover:bg-accent/90 transition"
                            >
                                <ArrowLeft size={13} />
                                <span>Return to Dataset Management</span>
                            </button>
                        </div>
                    ) : (
                        <ModelOrchestrationPanel
                            datasetId={Number(datasetId)}
                            policy={policy}
                            labelsById={labelsById}
                            catalog={catalog}
                            onSavePolicy={handleSavePolicy}
                            onDeletePolicy={handleDeletePolicy}
                            isSaving={isSaving}
                            isDeleting={isDeleting}
                            canEdit={canEdit}
                        />
                    )}
                </main>
            </div>
        </DatasetManagementLayout>
    );
}
