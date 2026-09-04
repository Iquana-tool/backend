import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Clock, Loader2, Plus, Wand2 } from "lucide-react";
import DatasetManagementLayout from "../components/datasets/gallery/DatasetManagementLayout";
import { useDataset } from "../contexts/DatasetContext";
import LabelModelPlanner from "../components/inference/LabelModelPlanner";
import WriteModeSelector from "../components/inference/WriteModeSelector";
import ReplaceWarningModal from "../components/inference/ReplaceWarningModal";
import InferenceProgressPanel, {
    STATUS_STYLE,
    TERMINAL_JOB_STATUSES,
} from "../components/inference/InferenceProgressPanel";
import {
    cancelInferenceJob,
    deleteInferenceJob,
    fetchLabels,
    getInferenceJobs,
    getInferenceModelCatalog,
    getInferenceRoutingPolicy,
    getInferenceScopeCounts,
    previewInferenceReplace,
    startInferenceJob,
    streamInferenceJob,
} from "../api";
import { BATCH_INFERENCE_TASKS } from "../constants/tasks";
import { getBatchStepsFromPolicy } from "../utils/inferenceRouting";

/**
 * Batch inference: annotate a whole dataset without opening the canvas.
 *
 * The page is the plan editor plus the progress view for one run at a time, laid out like the
 * training page (history rail, config or progress on the right) so the two AI pages behave
 * the same way. Dataset routing is a read-only source for preselecting the plan.
 */

const DEFAULT_OPTIONS = {
    write_mode: "patch",
    nms_iou: 0.7,
    preserve_reviewed: true,
    unparented: "drop",
    min_parent_containment: 0.5,
};

const SCOPE_OPTIONS = [
    { value: "all", label: "Every image", countKey: "total" },
    { value: "not_started", label: "Not annotated yet", countKey: "not_started" },
    { value: "unreviewed", label: "Not fully reviewed", countKey: "unreviewed" },
];

const RUN_NAME_PATTERN = /^[\p{L}\p{N}_\-\s]{1,80}$/u;

const LEGACY_STEP_KEYS = new Set(["retrieval_strategy", "top_k"]);

// Retrieval fields remain local mirrors for old persisted plans, but canonical requests carry
// conditioning through inputs. min_confidence is different: it is a gateway-owned post-filter
// and remains valid alongside canonical model inputs.
const stepsForRequest = (steps) =>
    steps.map((step) =>
        step.inputs
            ? Object.fromEntries(
                  Object.entries(step).filter(([key]) => !LEGACY_STEP_KEYS.has(key))
              )
            : step
    );

const formatTime = (value) => (value ? new Date(value).toLocaleString() : "—");

function JobCard({ job, selected, onClick }) {
    const percent = job.total_units
        ? Math.round((job.done_units / job.total_units) * 100)
        : 0;
    return (
        <button
            onClick={onClick}
            className={`w-full text-left p-3 rounded-xl border transition-colors ${
                selected ? "border-acLn bg-acS" : "border-ln bg-p1 hover:bg-hv"
            }`}
        >
            <div className="flex items-center justify-between mb-1">
                <span
                    className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${
                        STATUS_STYLE[job.status] || STATUS_STYLE.pending
                    }`}
                >
                    {job.status}
                </span>
                <span className="text-[11px] text-t3 flex items-center gap-1">
                    <Clock size={11} /> {formatTime(job.created_at)}
                </span>
            </div>
            {job.name && (
                <p className="text-xs font-medium text-t1 truncate mb-0.5">{job.name}</p>
            )}
            <p className="text-xs text-t2">
                {job.steps?.length ?? 0} label{(job.steps?.length ?? 0) === 1 ? "" : "s"} ·{" "}
                {job.image_count?.toLocaleString() ?? 0} images · {percent}%
            </p>
            {job.write_mode === "replace" && (
                <p className="text-[11px] text-err mt-0.5">replace</p>
            )}
        </button>
    );
}

export default function BatchInferencePage() {
    const { datasetId } = useParams();
    const { currentDataset } = useDataset();
    const navigate = useNavigate();

    const [labelsById, setLabelsById] = useState({});
    const [catalog, setCatalog] = useState({ models: [], retrieval_strategies: [] });
    const [scope, setScope] = useState({ total: 0, not_started: 0, unreviewed: 0 });
    const [policy, setPolicy] = useState(null);
    const [appliedFilter, setAppliedFilter] = useState(null);

    const [stepsByLabel, setStepsByLabel] = useState({});
    const [options, setOptions] = useState(DEFAULT_OPTIONS);
    const [imageSelection, setImageSelection] = useState("all");
    const [runName, setRunName] = useState("");

    const [jobs, setJobs] = useState([]);
    const [mode, setMode] = useState("config"); // "config" | "run"
    const [selectedJob, setSelectedJob] = useState(null);
    const [activeJobId, setActiveJobId] = useState(null);

    const [replacePreview, setReplacePreview] = useState(null);
    const [showReplaceWarning, setShowReplaceWarning] = useState(false);
    const [isStarting, setIsStarting] = useState(false);
    const [isCancelling, setIsCancelling] = useState(false);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const streamRef = useRef(null);

    const steps = useMemo(
        () => Object.values(stepsByLabel).filter(Boolean),
        [stepsByLabel]
    );

    const requestBody = useCallback(
        (confirmReplace = false) => ({
            dataset_id: Number(datasetId),
            name: runName.trim() || undefined,
            steps: stepsForRequest(steps),
            image_selection: imageSelection,
            options,
            confirm_replace: confirmReplace,
        }),
        [datasetId, runName, steps, imageSelection, options]
    );

    const loadJobs = useCallback(async (isCancelled = () => false) => {
        try {
            const list = await getInferenceJobs(datasetId);
            if (!isCancelled()) {
                setJobs(Array.isArray(list) ? list : []);
            }
            return list;
        } catch (e) {
            // Non-fatal: the history rail simply stays empty.
            if (!isCancelled()) setJobs([]);
            return [];
        }
    }, [datasetId]);

    const applyPolicyToBatchSteps = useCallback(
        (bindings, availableLabels, availableCatalog, preferredTask = null) => {
            if (!Array.isArray(bindings)) return;
            setStepsByLabel(
                getBatchStepsFromPolicy(
                    { bindings },
                    availableLabels,
                    availableCatalog?.models || [],
                    preferredTask
                )
            );
        },
        []
    );

    const handleApplyFilter = useCallback(
        (filterKey) => {
            if (!policy?.bindings) return;
            applyPolicyToBatchSteps(
                policy.bindings,
                labelsById,
                catalog,
                filterKey === "all" ? null : filterKey
            );
            setAppliedFilter(filterKey);
        },
        [policy, labelsById, catalog, applyPolicyToBatchSteps]
    );

    useEffect(() => {
        if (!datasetId) return undefined;
        let cancelled = false;

        setIsLoading(true);
        setError(null);
        setPolicy(null);
        setAppliedFilter(null);
        setStepsByLabel({});
        setOptions(DEFAULT_OPTIONS);
        setJobs([]);
        setSelectedJob(null);
        setActiveJobId(null);
        setMode("config");

        (async () => {
            try {
                const [labelResponse, catalogResponse, scopeResponse, policyResponse] =
                    await Promise.all([
                        fetchLabels(datasetId),
                        getInferenceModelCatalog(datasetId),
                        getInferenceScopeCounts(datasetId),
                        getInferenceRoutingPolicy(datasetId).catch(() => null),
                    ]);
                if (cancelled) return;

                const loadedLabels = labelResponse?.labels?.id_to_label_object || {};
                const loadedCatalog = catalogResponse || {
                    models: [],
                    retrieval_strategies: [],
                };

                setLabelsById(loadedLabels);
                setCatalog(loadedCatalog);
                setScope(scopeResponse || { total: 0, not_started: 0, unreviewed: 0 });
                setPolicy(policyResponse || null);
                setAppliedFilter(null);

                const bindings = Array.isArray(policyResponse?.bindings)
                    ? policyResponse.bindings
                    : [];
                const hasInstance = bindings.some(
                    (binding) => binding.task === "instance-segmentation"
                );
                const hasCrossImage = bindings.some(
                    (binding) => binding.task === "cross-image-suggestion"
                );
                if (hasInstance !== hasCrossImage) {
                    const task = hasInstance
                        ? "instance-segmentation"
                        : "cross-image-suggestion";
                    applyPolicyToBatchSteps(bindings, loadedLabels, loadedCatalog, task);
                    setAppliedFilter(task);
                }
            } catch (e) {
                if (!cancelled) {
                    setError(e.message || "Could not load the models for this dataset.");
                }
            } finally {
                if (!cancelled) setIsLoading(false);
            }
            // Landing while a run is going should show that run, not an empty config form —
            // this is the tab somebody comes back to in order to check on it.
            if (cancelled) return;
            const list = await loadJobs(() => cancelled);
            if (cancelled) return;
            const running = (list || []).find((job) => !TERMINAL_JOB_STATUSES.has(job.status));
            if (running) {
                setSelectedJob(running);
                setActiveJobId(running.id);
                setMode("run");
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [datasetId, loadJobs, applyPolicyToBatchSteps]);

    // Stream the active run; tear the stream down on change/unmount.
    useEffect(() => {
        if (!activeJobId) return undefined;
        const controller = streamInferenceJob(
            activeJobId,
            (snapshot) => {
                setSelectedJob(snapshot);
                setJobs((current) =>
                    current.some((job) => job.id === snapshot.id)
                        ? current.map((job) => (job.id === snapshot.id ? snapshot : job))
                        : [snapshot, ...current]
                );
                if (TERMINAL_JOB_STATUSES.has(snapshot.status)) {
                    setActiveJobId(null);
                    loadJobs();
                }
            },
            (streamError) => setError(streamError.message || "Lost the progress stream.")
        );
        streamRef.current = controller;
        return () => controller.abort();
    }, [activeJobId, loadJobs]);

    // Keep the replace preview in step with the scope and the reviewed-work toggle: the
    // warning is only useful if its numbers describe the run that is actually configured.
    useEffect(() => {
        if (options.write_mode !== "replace" || steps.length === 0) {
            setReplacePreview(null);
            return undefined;
        }
        let cancelled = false;
        previewInferenceReplace(requestBody())
            .then((preview) => {
                if (!cancelled) setReplacePreview(preview);
            })
            .catch(() => {
                if (!cancelled) setReplacePreview(null);
            });
        return () => {
            cancelled = true;
        };
    }, [options.write_mode, options.preserve_reviewed, imageSelection, steps.length, requestBody]);

    const setStep = (labelId, step) => {
        setAppliedFilter(null);
        setStepsByLabel((current) => ({ ...current, [labelId]: step }));
    };

    const start = async (confirmReplace) => {
        setError(null);
        setIsStarting(true);
        try {
            const snapshot = await startInferenceJob(requestBody(confirmReplace));
            setShowReplaceWarning(false);
            setJobs((current) => [snapshot, ...current]);
            setSelectedJob(snapshot);
            setActiveJobId(snapshot.id);
            setMode("run");
        } catch (e) {
            setError(e.message || "Could not start the run.");
        } finally {
            setIsStarting(false);
        }
    };

    const handleStart = () => {
        if (options.write_mode === "replace") {
            setShowReplaceWarning(true);
            return;
        }
        start(false);
    };

    const handleCancel = async () => {
        if (!selectedJob) return;
        setIsCancelling(true);
        try {
            setSelectedJob(await cancelInferenceJob(selectedJob.id));
        } catch (e) {
            setError(e.message || "Could not stop the run.");
        } finally {
            setIsCancelling(false);
        }
    };

    const handleDelete = async () => {
        if (!selectedJob) return;
        try {
            await deleteInferenceJob(selectedJob.id);
            setJobs((current) => current.filter((job) => job.id !== selectedJob.id));
            setSelectedJob(null);
            setActiveJobId(null);
            setMode("config");
        } catch (e) {
            setError(e.message || "Could not remove the run.");
        }
    };

    const handleSelectJob = (job) => {
        setMode("run");
        setSelectedJob(job);
        setActiveJobId(TERMINAL_JOB_STATUSES.has(job.status) ? null : job.id);
    };

    const batchEligibleBindings = useMemo(
        () =>
            Array.isArray(policy?.bindings)
                ? policy.bindings.filter((binding) =>
                      BATCH_INFERENCE_TASKS.includes(binding.task)
                  )
                : [],
        [policy]
    );
    const hasInstanceRoutes = batchEligibleBindings.some(
        (binding) => binding.task === "instance-segmentation"
    );
    const hasCrossImageRoutes = batchEligibleBindings.some(
        (binding) => binding.task === "cross-image-suggestion"
    );

    const runNameError =
        runName.length > 0 && !RUN_NAME_PATTERN.test(runName)
            ? "Use letters, numbers, spaces, hyphens or underscores (max 80)."
            : null;
    const scopeCount = scope[SCOPE_OPTIONS.find((o) => o.value === imageSelection).countKey] ?? 0;
    const hasActiveRun = jobs.some((job) => !TERMINAL_JOB_STATUSES.has(job.status));
    const canStart =
        steps.length > 0 && scopeCount > 0 && !hasActiveRun && !runNameError && !isStarting;

    return (
        <DatasetManagementLayout>
            <div className="h-full flex flex-col bg-p1 overflow-hidden">
                <div className="bg-p1 border-b border-ln px-4 sm:px-6 lg:px-8 py-4 flex items-center gap-3">
                    <Wand2 className="w-6 h-6 text-ac" />
                    <div>
                        <h1 className="text-2xl font-bold text-t1">Batch Inference</h1>
                        <p className="text-sm text-t2">
                            Let your models annotate{" "}
                            {currentDataset?.name ? `“${currentDataset.name}”` : "this dataset"} —
                            one model per label, run in hierarchy order.
                        </p>
                    </div>
                </div>

                <div className="flex-1 flex overflow-hidden">
                    <aside className="w-72 shrink-0 border-r border-ln flex flex-col">
                        <div className="p-3 border-b border-ln">
                            <button
                                onClick={() => {
                                    setMode("config");
                                    setSelectedJob(null);
                                }}
                                className={`w-full inline-flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                                    mode === "config"
                                        ? "bg-accent text-onAccent"
                                        : "bg-acS text-ac hover:bg-acS"
                                }`}
                            >
                                <Plus size={16} /> New Run
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-3 space-y-2">
                            <p className="text-[11px] font-semibold uppercase tracking-wide text-t3 px-1">
                                Run history
                            </p>
                            {jobs.length === 0 && <p className="text-xs text-t3 px-1">No runs yet.</p>}
                            {jobs.map((job) => (
                                <JobCard
                                    key={job.id}
                                    job={job}
                                    selected={mode === "run" && selectedJob?.id === job.id}
                                    onClick={() => handleSelectJob(job)}
                                />
                            ))}
                        </div>
                    </aside>

                    <main className="flex-1 overflow-y-auto p-6">
                        {error && (
                            <div className="mb-4 p-3 bg-errBg border border-errLn rounded-lg text-sm text-err">
                                {error}
                            </div>
                        )}

                        {mode === "run" && selectedJob ? (
                            <div className="max-w-3xl">
                                <InferenceProgressPanel
                                    job={selectedJob}
                                    onCancel={handleCancel}
                                    isCancelling={isCancelling}
                                    onReview={() =>
                                        navigate(`/dataset/${datasetId}/review`, {
                                            state: { onlySubmitted: false },
                                        })
                                    }
                                    onDelete={handleDelete}
                                    onOpenImage={(imageId) =>
                                        navigate(`/dataset/${datasetId}/annotate/${imageId}`)
                                    }
                                />
                            </div>
                        ) : isLoading ? (
                            <p className="flex items-center gap-2 text-sm text-t3">
                                <Loader2 className="w-4 h-4 animate-spin" /> Loading models and
                                labels…
                            </p>
                        ) : (
                            <div className="max-w-3xl space-y-8">
                                <section>
                                    <h2 className="text-base font-semibold text-t1 mb-1">
                                        1. Which images
                                    </h2>
                                    <div className="flex flex-wrap gap-2">
                                        {SCOPE_OPTIONS.map((option) => (
                                            <button
                                                key={option.value}
                                                type="button"
                                                onClick={() => setImageSelection(option.value)}
                                                aria-pressed={imageSelection === option.value}
                                                className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                                                    imageSelection === option.value
                                                        ? "border-acLn bg-acS text-ac"
                                                        : "border-ln bg-p1 text-t2 hover:bg-hv"
                                                }`}
                                            >
                                                {option.label}
                                                <span className="ml-1.5 text-[11px] text-t3">
                                                    {(scope[option.countKey] ?? 0).toLocaleString()}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                </section>

                                <section>
                                    <div className="flex flex-wrap items-start justify-between gap-3 mb-1">
                                        <div>
                                            <h2 className="text-base font-semibold text-t1">
                                                2. Which model annotates which label
                                            </h2>
                                            <p className="text-xs text-t3">
                                                Bind a model to each label you want annotated. Labels
                                                left on “Skip” are untouched.
                                            </p>
                                        </div>
                                        <Link
                                            to={`/dataset/${datasetId}/model-orchestration`}
                                            className="text-xs font-medium text-ac hover:underline"
                                        >
                                            Edit model routing
                                        </Link>
                                    </div>

                                    {batchEligibleBindings.length > 0 && (
                                        <div className="flex flex-wrap items-center gap-2 mb-3">
                                            <button
                                                type="button"
                                                onClick={() => handleApplyFilter("all")}
                                                aria-pressed={appliedFilter === "all"}
                                                className={`px-2.5 py-1 text-xs rounded-lg border ${
                                                    appliedFilter === "all"
                                                        ? "border-acLn bg-acS text-ac"
                                                        : "border-ln bg-p1 text-t2 hover:bg-hv"
                                                }`}
                                            >
                                                Apply All Routes
                                            </button>
                                            {hasInstanceRoutes && hasCrossImageRoutes && (
                                                <>
                                                    <button
                                                        type="button"
                                                        onClick={() =>
                                                            handleApplyFilter(
                                                                "instance-segmentation"
                                                            )
                                                        }
                                                        aria-pressed={
                                                            appliedFilter ===
                                                            "instance-segmentation"
                                                        }
                                                        className={`px-2.5 py-1 text-xs rounded-lg border ${
                                                            appliedFilter ===
                                                            "instance-segmentation"
                                                                ? "border-acLn bg-acS text-ac"
                                                                : "border-ln bg-p1 text-t2 hover:bg-hv"
                                                        }`}
                                                    >
                                                        Instance Seg Only
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={() =>
                                                            handleApplyFilter(
                                                                "cross-image-suggestion"
                                                            )
                                                        }
                                                        aria-pressed={
                                                            appliedFilter ===
                                                            "cross-image-suggestion"
                                                        }
                                                        className={`px-2.5 py-1 text-xs rounded-lg border ${
                                                            appliedFilter ===
                                                            "cross-image-suggestion"
                                                                ? "border-acLn bg-acS text-ac"
                                                                : "border-ln bg-p1 text-t2 hover:bg-hv"
                                                        }`}
                                                    >
                                                        Cross-Image Only
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    )}

                                    <LabelModelPlanner
                                        labelsById={labelsById}
                                        models={catalog.models}
                                        strategies={catalog.retrieval_strategies}
                                        steps={steps}
                                        onChange={setStep}
                                        allowedTasks={BATCH_INFERENCE_TASKS}
                                    />
                                </section>

                                <section>
                                    <h2 className="text-base font-semibold text-t1 mb-1">
                                        3. What happens to existing annotations
                                    </h2>
                                    <WriteModeSelector
                                        options={options}
                                        onChange={setOptions}
                                        replacePreview={replacePreview}
                                    />
                                </section>

                                {steps.some((step) => (labelsById[step.label_id]?.parent_id ?? null) !== null) && (
                                    <section>
                                        <h2 className="text-base font-semibold text-t1 mb-1">
                                            4. Nested objects
                                        </h2>
                                        <label className="flex items-center gap-2 text-sm text-t2">
                                            <input
                                                type="checkbox"
                                                checked={options.unparented === "keep_at_root"}
                                                onChange={(event) =>
                                                    setOptions({
                                                        ...options,
                                                        unparented: event.target.checked
                                                            ? "keep_at_root"
                                                            : "drop",
                                                    })
                                                }
                                                className="h-4 w-4"
                                            />
                                            Keep predictions that fall outside every parent object
                                        </label>
                                        <p className="text-[11px] text-t3 mt-1">
                                            Off by default: a nested object predicted outside any
                                            parent is usually a false positive.
                                        </p>
                                    </section>
                                )}

                                <section>
                                    <label
                                        htmlFor="inference-run-name"
                                        className="block text-sm font-medium text-t1 mb-1"
                                    >
                                        Run name <span className="text-t3 font-normal">(optional)</span>
                                    </label>
                                    <input
                                        id="inference-run-name"
                                        type="text"
                                        value={runName}
                                        onChange={(event) => setRunName(event.target.value)}
                                        maxLength={80}
                                        aria-invalid={Boolean(runNameError)}
                                        placeholder="e.g. Cells + nuclei, first pass"
                                        className={`w-full px-3 py-2 text-sm border rounded-lg bg-well text-t1 focus:ring-2 focus:ring-ac focus:border-transparent ${
                                            runNameError ? "border-errLn" : "border-ln"
                                        }`}
                                    />
                                    {runNameError && (
                                        <p className="text-[11px] text-err mt-1">{runNameError}</p>
                                    )}
                                </section>

                                {hasActiveRun && (
                                    <p className="p-3 bg-warnBg border border-warnLn rounded-lg text-sm text-warn">
                                        A run is already going on this dataset. Wait for it to finish
                                        or stop it first — two runs writing the same images would
                                        fight over duplicates.
                                    </p>
                                )}

                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={handleStart}
                                        disabled={!canStart}
                                        className={`inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-onAccent rounded-lg hover:brightness-110 transition-colors disabled:opacity-60 ${
                                            options.write_mode === "replace" ? "bg-err" : "bg-accent"
                                        }`}
                                    >
                                        {isStarting ? (
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                        ) : (
                                            <Wand2 className="w-4 h-4" />
                                        )}
                                        {isStarting ? "Starting…" : "Start Inference"}
                                    </button>
                                    {steps.length > 0 && scopeCount > 0 && (
                                        <span className="text-xs text-t3">
                                            {(steps.length * scopeCount).toLocaleString()} model runs
                                            ({steps.length} label{steps.length === 1 ? "" : "s"} ×{" "}
                                            {scopeCount.toLocaleString()} images)
                                        </span>
                                    )}
                                    {steps.length === 0 && (
                                        <span className="text-xs text-t3">
                                            Bind at least one label to a model.
                                        </span>
                                    )}
                                </div>
                            </div>
                        )}
                    </main>
                </div>
            </div>

            <ReplaceWarningModal
                isOpen={showReplaceWarning}
                preview={replacePreview}
                preserveReviewed={options.preserve_reviewed}
                imageCount={scopeCount}
                isStarting={isStarting}
                onCancel={() => setShowReplaceWarning(false)}
                onConfirm={() => start(true)}
            />
        </DatasetManagementLayout>
    );
}
