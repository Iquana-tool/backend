import React, { useEffect, useState } from "react";
import {
    Activity,
    AlertTriangle,
    CheckCircle2,
    ClipboardCheck,
    Cpu,
    Layers,
    Loader2,
    Scissors,
    StopCircle,
    Trash2,
    Unlink,
} from "lucide-react";
import { getInferenceJobItems } from "../../api/inference";

/** Statuses after which nothing moves again. Mirrors TERMINAL_JOB_STATUSES on the backend. */
export const TERMINAL_JOB_STATUSES = new Set(["succeeded", "partial", "failed", "cancelled"]);

export const STATUS_STYLE = {
    pending: "bg-well text-t2",
    running: "bg-acS text-ac",
    cancelling: "bg-warnBg text-warn",
    cancelled: "bg-warnBg text-warn",
    succeeded: "bg-okBg text-ok",
    partial: "bg-warnBg text-warn",
    failed: "bg-errBg text-err",
};

/** "2h 5m" / "4m 20s" / "18s" — an ETA nobody has to parse. */
export const formatDuration = (seconds) => {
    if (seconds == null || !Number.isFinite(seconds)) return null;
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${total % 60}s`;
    return `${total}s`;
};

function Stat({ icon: Icon, value, label, tone = "text-t1" }) {
    return (
        <div className="p-3 rounded-lg border border-ln bg-well">
            <div className="flex items-center gap-1.5 text-[11px] text-t3">
                <Icon size={12} /> {label}
            </div>
            <p className={`text-lg font-semibold mt-0.5 ${tone}`}>{(value ?? 0).toLocaleString()}</p>
        </div>
    );
}

/**
 * Live view of one run.
 *
 * The overall bar counts *work units* (one model on one image), not images: a plan with three
 * labels does three times the work of a plan with one, and a bar that ignored that would sit
 * at 100% for two thirds of the run. The per-level bars underneath show the hierarchy
 * actually being respected — level 2 stays empty until level 1 is full.
 */
export default function InferenceProgressPanel({
    job,
    onCancel,
    isCancelling,
    onReview,
    onDelete,
    onOpenImage,
}) {
    const [failedItems, setFailedItems] = useState([]);

    const total = job.total_units || 0;
    const done = job.done_units || 0;
    const percent = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    const isActive = !TERMINAL_JOB_STATUSES.has(job.status);
    const eta = formatDuration(job.eta_seconds);

    useEffect(() => {
        if (!job.failed_units) {
            setFailedItems([]);
            return undefined;
        }
        let cancelled = false;
        getInferenceJobItems(job.id, "failed")
            .then((items) => {
                if (!cancelled) setFailedItems(Array.isArray(items) ? items : []);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [job.id, job.failed_units]);

    return (
        <div className="space-y-5">
            <div className="flex flex-wrap items-center gap-3">
                <span
                    className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                        STATUS_STYLE[job.status] || STATUS_STYLE.pending
                    }`}
                >
                    {job.status}
                </span>
                <span className="text-sm text-t2">
                    {done.toLocaleString()} / {total.toLocaleString()} steps
                    {job.image_count ? ` · ${job.image_count.toLocaleString()} images` : ""}
                </span>
                {isActive && eta && <span className="text-sm text-t3">~{eta} left</span>}
                {job.write_mode === "replace" && (
                    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-errBg text-err">
                        replace
                    </span>
                )}
            </div>

            <div>
                <div className="w-full bg-hv2 rounded h-2.5 overflow-hidden">
                    <div
                        className="bg-accent h-2.5 rounded"
                        style={{ width: `${percent}%`, transition: "width 0.5s" }}
                    />
                </div>
                <p className="text-[11px] text-t3 mt-1">{percent}% complete</p>
            </div>

            {job.status === "running" && job.current_step && (
                <p className="flex items-center gap-2 text-sm text-t2">
                    <Loader2 className="w-4 h-4 animate-spin text-ac" />
                    Level {job.current_step.level + 1} · {job.current_step.label_name} ·{" "}
                    {job.current_step.model_name}
                </p>
            )}

            {job.levels?.length > 1 && (
                <div className="space-y-2">
                    <h3 className="flex items-center gap-1.5 text-sm font-semibold text-t1">
                        <Layers size={14} /> Hierarchy levels
                    </h3>
                    <p className="text-[11px] text-t3">
                        Each level finishes across the whole dataset before the next one starts, so
                        nested predictions always have a parent to attach to.
                    </p>
                    {job.levels.map((level) => {
                        const levelPercent = level.total
                            ? Math.round((level.done / level.total) * 100)
                            : 0;
                        return (
                            <div key={level.level}>
                                <div className="flex items-center justify-between text-[11px] text-t2 mb-0.5">
                                    <span>
                                        Level {level.level + 1}
                                        {level.label_names.length
                                            ? ` · ${level.label_names.join(", ")}`
                                            : ""}
                                    </span>
                                    <span className="text-t3">
                                        {level.done}/{level.total}
                                    </span>
                                </div>
                                <div className="w-full bg-hv2 rounded h-1.5">
                                    <div
                                        className="bg-accent h-1.5 rounded"
                                        style={{ width: `${levelPercent}%`, transition: "width 0.5s" }}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {job.recent_activity?.length > 0 && (
                <div>
                    <h3 className="flex items-center gap-1.5 text-sm font-semibold text-t1 mb-1">
                        <Activity size={14} /> Just annotated
                    </h3>
                    <p className="text-[11px] text-t3 mb-2">
                        Click an image to open it and check the predictions.
                    </p>
                    <div className="max-h-52 overflow-y-auto border border-ln rounded-lg divide-y divide-ln">
                        {job.recent_activity.map((entry) => (
                            <button
                                key={`${entry.image_id}-${entry.finished_at}`}
                                onClick={() => onOpenImage?.(entry.image_id)}
                                className="w-full flex items-center justify-between gap-3 px-3 py-2 text-xs text-left hover:bg-hv"
                            >
                                <span className="min-w-0 flex-1 truncate text-t1">
                                    {entry.image_name || `Image ${entry.image_id}`}
                                    <span className="text-t3"> · {entry.label_name}</span>
                                </span>
                                <span className="shrink-0 text-t2">
                                    <span className={entry.contours_created ? "text-ok font-medium" : "text-t3"}>
                                        +{entry.contours_created}
                                    </span>
                                    {entry.contours_suppressed > 0 && (
                                        <span className="text-t3"> · {entry.contours_suppressed} dup</span>
                                    )}
                                </span>
                            </button>
                        ))}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Stat icon={CheckCircle2} label="Objects added" value={job.contours_created} tone="text-ok" />
                <Stat icon={Scissors} label="Duplicates dropped" value={job.contours_suppressed} />
                <Stat icon={Unlink} label="Without a parent" value={job.contours_unparented} />
                {job.write_mode === "replace" ? (
                    <Stat icon={Trash2} label="Deleted first" value={job.contours_deleted} tone="text-err" />
                ) : (
                    <Stat icon={Cpu} label="Steps failed" value={job.failed_units} tone={job.failed_units ? "text-err" : "text-t1"} />
                )}
            </div>

            {job.error && (
                <div className="p-3 rounded-lg border border-errLn bg-errBg text-sm text-err">
                    {job.error}
                </div>
            )}

            {failedItems.length > 0 && (
                <div>
                    <h3 className="flex items-center gap-1.5 text-sm font-semibold text-t1 mb-1">
                        <AlertTriangle size={14} className="text-warn" /> {failedItems.length} failed
                        step{failedItems.length === 1 ? "" : "s"}
                    </h3>
                    <p className="text-[11px] text-t3 mb-2">
                        A failed step does not stop the run — the remaining images were still
                        processed.
                    </p>
                    <div className="max-h-52 overflow-y-auto border border-ln rounded-lg divide-y divide-ln">
                        {failedItems.map((item) => (
                            <div key={item.id} className="px-3 py-2 text-xs">
                                <p className="text-t1 font-medium truncate">
                                    {item.image_name || `Image ${item.image_id}`} · {item.label_name}
                                </p>
                                <p className="text-t3 break-words">{item.error}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="flex flex-wrap gap-2">
                {isActive && (
                    <button
                        onClick={onCancel}
                        disabled={isCancelling || job.status === "cancelling"}
                        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-onAccent bg-err rounded-lg hover:brightness-110 disabled:opacity-60"
                    >
                        {isCancelling ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <StopCircle className="w-4 h-4" />
                        )}
                        {job.status === "cancelling" ? "Stopping after this image…" : "Stop run"}
                    </button>
                )}
                {!isActive && job.contours_created > 0 && (
                    <button
                        onClick={onReview}
                        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-onAccent bg-accent rounded-lg hover:brightness-110"
                    >
                        <ClipboardCheck className="w-4 h-4" />
                        Review the predictions
                    </button>
                )}
                {/* Always reachable except while a worker is mid-image: a run that failed to
                    queue, or is stuck asking to be cancelled, has to be removable or it
                    blocks every future run on this dataset. */}
                {job.status !== "running" && (
                    <button
                        onClick={onDelete}
                        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-t2 bg-p1 border border-ln rounded-lg hover:bg-hv"
                    >
                        <Trash2 className="w-4 h-4" />
                        Remove from history
                    </button>
                )}
            </div>

            {!isActive && job.contours_created > 0 && (
                <p className="text-[11px] text-t3">
                    Predicted objects are unreviewed, so they are already waiting in the review
                    queue.
                </p>
            )}
        </div>
    );
}
