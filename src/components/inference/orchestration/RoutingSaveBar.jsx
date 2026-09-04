import React, { useState } from "react";
import {
  Bookmark,
  RotateCcw,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Check,
} from "lucide-react";

/**
 * RoutingSaveBar component
 *
 * Routing policy save and action bar matching the target design:
 * Left: Amber status dot, unsaved changes count, and diff preview summary.
 * Right: Clear policy, Reset changes, and teal Save routing policy button.
 */
export default function RoutingSaveBar({
  hasUnsavedChanges = false,
  changeSummary = { totalCount: 0, items: [], summaryText: "" },
  hasSavedPolicy = false,
  onSave,
  onReset,
  onClear,
  isSaving = false,
  isDeleting = false,
  canEdit = true,
  statusMessage = null,
  className = "",
}) {
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const handleClearClick = () => {
    if (!canEdit || isSaving || isDeleting) return;
    setShowClearConfirm(true);
  };

  const handleConfirmClear = async () => {
    setShowClearConfirm(false);
    if (onClear) {
      await onClear();
    }
  };

  const handleCancelClear = () => {
    setShowClearConfirm(false);
  };

  return (
    <aside
      aria-label="Routing policy save bar"
      data-testid="routing-save-bar"
      className={`w-full rounded-2xl border border-slate-800 bg-[#151b23] transition-all mt-6 shadow-xs ${
        hasUnsavedChanges ? "py-3 px-6 ring-1 ring-amber-500/20 shadow-md" : "py-2.5 px-6"
      } ${className}`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 min-h-[38px]">
        {/* Left Side: Status / Change diff preview */}
        <div className="min-w-0 flex-1 flex items-center gap-2.5">
          {statusMessage ? (
            <div className="flex items-center gap-2 text-xs">
              {statusMessage.type === "success" ? (
                <CheckCircle2 size={15} className="text-emerald-500 shrink-0" />
              ) : (
                <AlertTriangle size={15} className="text-err shrink-0" />
              )}
              <span
                className={
                  statusMessage.type === "success"
                    ? "text-emerald-400 font-medium"
                    : "text-err font-medium"
                }
              >
                {statusMessage.text}
              </span>
            </div>
          ) : hasUnsavedChanges ? (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" />
              <div className="min-w-0 flex flex-wrap items-baseline gap-1.5 text-xs">
                <span className="font-semibold text-t1 shrink-0">
                  {changeSummary.totalCount} unsaved{" "}
                  {changeSummary.totalCount === 1 ? "change" : "changes"}
                </span>
                {changeSummary.summaryText && (
                  <span
                    className="text-t3 truncate max-w-xl font-normal"
                    title={changeSummary.summaryText}
                  >
                    · {changeSummary.summaryText}
                  </span>
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center gap-2 text-xs text-t3">
              {hasSavedPolicy ? (
                <>
                  <CheckCircle2 size={14} className="text-emerald-500 shrink-0" />
                  <span className="text-t2 font-medium">All routing changes saved</span>
                </>
              ) : (
                <span>No custom routing policy saved</span>
              )}
            </div>
          )}
        </div>

        {/* Right Side: Actions (Clear, Reset, Save, or Read-only mode) */}
        <div className="flex items-center gap-2.5 shrink-0 self-end sm:self-center">
          {!canEdit ? (
            <span className="text-xs text-t3 font-medium">Read-only mode</span>
          ) : showClearConfirm ? (
            <div className="flex items-center gap-2 p-1 bg-errBg border border-errLn rounded-lg text-xs animate-in fade-in duration-150">
              <span className="text-err px-1.5 font-medium">Clear all custom routes?</span>
              <button
                type="button"
                onClick={handleConfirmClear}
                disabled={isDeleting || isSaving}
                className="px-2.5 py-1 bg-err text-white rounded font-medium hover:bg-err/90 transition text-xs disabled:opacity-50"
              >
                {isDeleting ? "Clearing…" : "Yes, Clear"}
              </button>
              <button
                type="button"
                onClick={handleCancelClear}
                disabled={isDeleting || isSaving}
                className="px-2 py-1 bg-well border border-ln rounded text-t2 hover:text-t1 transition text-xs"
              >
                Cancel
              </button>
            </div>
          ) : (
            <>
              {hasSavedPolicy && canEdit && (
                <button
                  type="button"
                  onClick={handleClearClick}
                  disabled={!canEdit || isSaving || isDeleting}
                  className="px-3 py-1.5 text-xs font-medium text-red-500 hover:text-red-400 transition disabled:opacity-50"
                  title="Clear all saved routing policies for this dataset"
                >
                  <span>{isDeleting ? "Clearing…" : "Clear policy"}</span>
                </button>
              )}

              {hasUnsavedChanges && canEdit && (
                <button
                  type="button"
                  onClick={onReset}
                  disabled={!canEdit || isSaving || isDeleting}
                  className="px-3 py-1.5 text-xs font-medium text-t2 hover:text-t1 bg-well border border-ln rounded-lg hover:bg-hv transition disabled:opacity-50"
                >
                  <span>Reset changes</span>
                </button>
              )}

              {canEdit && (
                <button
                  type="button"
                  onClick={onSave}
                  disabled={isSaving || isDeleting || !hasUnsavedChanges}
                  className={`inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg transition shadow-xs ${
                    hasUnsavedChanges
                      ? "bg-teal-500 hover:bg-teal-400 text-slate-950"
                      : "bg-well text-t3 border border-ln cursor-default"
                  } disabled:opacity-50`}
                >
                  {isSaving ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : hasUnsavedChanges ? (
                    <Bookmark size={13} className="fill-slate-950/20" />
                  ) : (
                    <Check size={13} className="text-emerald-500" />
                  )}
                  <span>
                    {isSaving
                      ? "Saving…"
                      : hasUnsavedChanges
                      ? "Save routing policy"
                      : "Saved"}
                  </span>
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
