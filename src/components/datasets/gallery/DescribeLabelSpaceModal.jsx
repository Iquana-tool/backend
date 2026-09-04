import React, { useEffect, useMemo, useState } from "react";
import {
  Sparkles,
  X,
  Send,
  Bot,
  ListTree,
  Wand2,
  Loader2,
  Trash2,
  ArrowLeft,
  Check,
} from "lucide-react";
import * as api from "../../../api";

/**
 * DescribeLabelSpaceModal
 *
 * A chat-style popup where the user describes, in plain language, what they
 * want to segment. The assistant turns that description into a draft
 * hierarchical label space (via an LLM on the backend), which the user reviews
 * and edits before it is persisted.
 *
 * Flow: "input" (describe) -> "review" (edit the generated draft) -> apply.
 * Generation requires the backend to be configured with an LLM key; otherwise
 * the dialog degrades gracefully to a "not available" notice and the user can
 * still build labels manually behind this dialog.
 */
const EXAMPLE_PROMPTS = [
  "Cells in a blood smear: whole cells, with their nuclei and granules inside them.",
  "Road scene objects: vehicles with their wheels and windows, pedestrians, and traffic signs.",
  "Plant anatomy: leaves with their veins, stems, and flowers with petals and stamen.",
];

// Deterministic color for a draft node (no DB id available yet).
const colorFor = (name) => {
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return `hsl(${Math.abs(hash) % 360}, 70%, 50%)`;
};

// Immutable update of a nested label node addressed by a path of indices.
const updateAtPath = (labels, path, updater) => {
  const [i, ...rest] = path;
  return labels.map((node, idx) => {
    if (idx !== i) return node;
    if (rest.length === 0) return updater(node);
    return { ...node, children: updateAtPath(node.children || [], rest, updater) };
  });
};

const removeAtPath = (labels, path) => {
  const [i, ...rest] = path;
  if (rest.length === 0) return labels.filter((_, idx) => idx !== i);
  return labels.map((node, idx) =>
    idx === i ? { ...node, children: removeAtPath(node.children || [], rest) } : node
  );
};

// Collect lowercased names to detect dataset-wide duplicates (the backend
// requires unique names, so we surface clashes before applying).
const collectNames = (labels, acc = []) => {
  labels.forEach((node) => {
    acc.push((node.name || "").trim().toLowerCase());
    collectNames(node.children || [], acc);
  });
  return acc;
};

const DraftTree = ({ labels, path = [], onRename, onRemove, duplicates }) =>
  labels.map((node, idx) => {
    const nodePath = [...path, idx];
    const key = nodePath.join("-");
    const trimmed = (node.name || "").trim().toLowerCase();
    const isDuplicate = trimmed && duplicates.has(trimmed);
    const isEmpty = !trimmed;
    return (
      <div key={key}>
        <div className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-hv">
          <div
            className="w-3.5 h-3.5 rounded-full shrink-0 ring-2 ring-p1 shadow-sm"
            style={{ backgroundColor: colorFor(node.name || "") }}
          />
          <input
            value={node.name}
            onChange={(e) => onRename(nodePath, e.target.value)}
            title={node.description || undefined}
            className={`min-w-0 flex-1 bg-transparent text-sm font-medium px-1.5 py-1 rounded border ${
              isDuplicate || isEmpty
                ? "border-errLn text-err focus:ring-err"
                : "border-transparent text-t1 hover:border-ln focus:border-transparent focus:ring-ac"
            } focus:ring-2 focus:outline-none`}
          />
          {node.children && node.children.length > 0 && (
            <span className="shrink-0 text-xs text-t3 bg-well px-2 py-0.5 rounded-full">
              {node.children.length} sub
            </span>
          )}
          <button
            onClick={() => onRemove(nodePath)}
            className="shrink-0 p-1 text-err hover:bg-errBg rounded opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
            title="Remove label"
          >
            <Trash2 size={14} />
          </button>
        </div>
        {node.children && node.children.length > 0 && (
          <div className="ml-[7px] pl-3 border-l border-ln">
            <DraftTree
              labels={node.children}
              path={nodePath}
              onRename={onRename}
              onRemove={onRemove}
              duplicates={duplicates}
            />
          </div>
        )}
      </div>
    );
  });

const DescribeLabelSpaceModal = ({ isOpen, onClose, dataset, onLabelsUpdated }) => {
  const [draftText, setDraftText] = useState("");
  const [stage, setStage] = useState("input"); // "input" | "review"
  const [draftLabels, setDraftLabels] = useState([]); // [{ name, description, children }]
  const [generating, setGenerating] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState(null);
  const [config, setConfig] = useState(null); // { enabled, model } | null
  const [refineText, setRefineText] = useState("");

  // Load availability whenever the dialog opens, and reset transient state.
  useEffect(() => {
    if (!isOpen) return;
    setStage("input");
    setDraftLabels([]);
    setError(null);
    setRefineText("");
    let cancelled = false;
    api
      .getLabelSpaceConfig()
      .then((cfg) => !cancelled && setConfig(cfg))
      .catch(() => !cancelled && setConfig({ enabled: false }));
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  // Duplicate / empty-name detection across the whole draft.
  const { duplicates, hasEmpty } = useMemo(() => {
    const names = collectNames(draftLabels);
    const seen = new Set();
    const dups = new Set();
    let empty = false;
    names.forEach((n) => {
      if (!n) {
        empty = true;
        return;
      }
      if (seen.has(n)) dups.add(n);
      seen.add(n);
    });
    return { duplicates: dups, hasEmpty: empty };
  }, [draftLabels]);

  if (!isOpen) return null;

  const enabled = config?.enabled ?? false;
  const canApply =
    draftLabels.length > 0 && duplicates.size === 0 && !hasEmpty && !applying && dataset?.id;

  const handleGenerate = async () => {
    if (!draftText.trim()) return;
    if (!enabled) return;
    setGenerating(true);
    setError(null);
    try {
      const result = await api.generateLabelSpace(draftText);
      setDraftLabels(result?.draft?.labels || []);
      setStage("review");
    } catch (err) {
      setError(err.message || "Failed to generate label space.");
    } finally {
      setGenerating(false);
    }
  };

  const handleRefine = async () => {
    if (!refineText.trim()) return;
    setGenerating(true);
    setError(null);
    try {
      const result = await api.refineLabelSpace(
        { labels: draftLabels },
        refineText,
        { description: draftText }
      );
      setDraftLabels(result?.draft?.labels || []);
      setRefineText("");
    } catch (err) {
      setError(err.message || "Failed to refine label space.");
    } finally {
      setGenerating(false);
    }
  };

  const handleApply = async () => {
    if (!canApply) return;
    setApplying(true);
    setError(null);
    try {
      await api.applyLabelSpace(dataset.id, { labels: draftLabels });
      if (onLabelsUpdated) await onLabelsUpdated();
      onClose();
    } catch (err) {
      setError(err.message || "Failed to apply label space.");
    } finally {
      setApplying(false);
    }
  };

  const handleRename = (path, value) =>
    setDraftLabels((labels) => updateAtPath(labels, path, (node) => ({ ...node, name: value })));
  const handleRemove = (path) => setDraftLabels((labels) => removeAtPath(labels, path));

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4 py-6">
        {/* Backdrop */}
        <div className="fixed inset-0 bg-scrim transition-opacity" onClick={onClose} />

        {/* Dialog */}
        <div className="relative inline-block w-full max-w-2xl text-left align-middle bg-p1 rounded-2xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="relative bg-p2 border-b border-ln px-6 py-5 text-t1">
            <button
              onClick={onClose}
              className="absolute top-4 right-4 p-1 rounded-lg text-t3 hover:text-t1 hover:bg-hv2 transition-colors"
              title="Close"
            >
              <X size={20} />
            </button>
            <div className="flex items-center gap-3 pr-8">
              <div className="flex items-center justify-center w-11 h-11 rounded-xl bg-hv shrink-0">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-bold">Describe your label space</h3>
                <p className="text-sm text-t3 mt-0.5">
                  {stage === "review"
                    ? "Review and tweak the draft — nothing is saved until you apply it."
                    : "Tell us what you want to segment and we'll draft the label hierarchy for you."}
                </p>
              </div>
            </div>
          </div>

          {/* Body */}
          <div className="px-6 py-5">
            {/* Error */}
            {error && (
              <div className="mb-4 p-3 bg-errBg border border-errLn rounded-lg text-sm text-err">
                {error}
              </div>
            )}

            {stage === "input" && (
              <>
                {/* Assistant intro message */}
                <div className="flex items-start gap-3 mb-5">
                  <div className="flex items-center justify-center w-9 h-9 rounded-full bg-acS text-ac shrink-0">
                    <Bot className="w-5 h-5" />
                  </div>
                  <div className="bg-well border border-ln rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-t2 leading-relaxed">
                    Describe the objects and categories in your images in plain language. I'll turn
                    them into a{" "}
                    <span className="inline-flex items-center gap-1 font-medium text-ac">
                      <ListTree className="w-3.5 h-3.5" />
                      nested label hierarchy
                    </span>{" "}
                    — nesting each part under the thing it is part of — that you can review
                    and tweak before applying.
                  </div>
                </div>

                {/* Not-configured notice */}
                {config && !enabled && (
                  <div className="mb-4 flex items-start gap-2 p-3 bg-warnBg border border-warnLn rounded-lg">
                    <Wand2 className="w-4 h-4 text-warn mt-0.5 shrink-0" />
                    <p className="text-xs text-warn leading-relaxed">
                      Automatic label-space generation isn't enabled on this server. Ask an
                      administrator to configure an LLM key, or build your hierarchy manually with the
                      controls behind this dialog.
                    </p>
                  </div>
                )}

                {/* Example prompts */}
                <div className="mb-4">
                  <p className="text-xs font-medium text-t3 mb-2">Try an example:</p>
                  <div className="flex flex-col gap-2">
                    {EXAMPLE_PROMPTS.map((example, index) => (
                      <button
                        key={index}
                        onClick={() => setDraftText(example)}
                        disabled={!enabled}
                        className="text-left text-sm text-t2 bg-p1 border border-ln rounded-lg px-3 py-2 hover:border-acLn hover:bg-acS transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Input */}
                <div className="relative">
                  <textarea
                    value={draftText}
                    onChange={(e) => setDraftText(e.target.value)}
                    rows={3}
                    disabled={!enabled || generating}
                    placeholder="e.g. I'm segmenting microscopy images of cells. I want to distinguish healthy from infected cells, and track their nuclei…"
                    className="w-full resize-none px-4 py-3 pr-12 border border-ln2 rounded-xl text-sm focus:ring-2 focus:ring-ac focus:border-transparent placeholder:text-t3 disabled:bg-well disabled:cursor-not-allowed"
                  />
                  <button
                    onClick={handleGenerate}
                    disabled={!enabled || generating || !draftText.trim()}
                    className="absolute bottom-3 right-3 flex items-center justify-center w-9 h-9 rounded-lg bg-accent text-onAccent hover:brightness-110 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Generate label space"
                  >
                    {generating ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                  </button>
                </div>

                <p className="text-[11px] text-t3 mt-2">
                  Generated hierarchies are drafts — you'll always be able to review and edit them
                  before anything is saved.
                </p>
              </>
            )}

            {stage === "review" && (
              <>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs font-medium text-t3">
                    Draft label space — rename or remove labels, then apply.
                  </p>
                  <button
                    onClick={() => setStage("input")}
                    className="inline-flex items-center gap-1 text-xs text-t3 hover:text-t1"
                  >
                    <ArrowLeft className="w-3.5 h-3.5" />
                    Edit description
                  </button>
                </div>

                <div className="max-h-72 overflow-y-auto border border-ln rounded-lg p-2 mb-3">
                  {draftLabels.length === 0 ? (
                    <p className="text-sm text-t3 text-center py-6">
                      No labels in this draft. Go back and refine your description.
                    </p>
                  ) : (
                    <DraftTree
                      labels={draftLabels}
                      onRename={handleRename}
                      onRemove={handleRemove}
                      duplicates={duplicates}
                    />
                  )}
                </div>

                {(duplicates.size > 0 || hasEmpty) && (
                  <div className="mb-3 p-2.5 bg-errBg border border-errLn rounded-lg text-xs text-err">
                    {hasEmpty && <div>Every label needs a name.</div>}
                    {duplicates.size > 0 && (
                      <div>
                        Label names must be unique across the dataset. Duplicates:{" "}
                        {Array.from(duplicates).join(", ")}.
                      </div>
                    )}
                  </div>
                )}

                {/* Refine */}
                <div className="relative">
                  <input
                    value={refineText}
                    onChange={(e) => setRefineText(e.target.value)}
                    disabled={generating}
                    onKeyDown={(e) => e.key === "Enter" && handleRefine()}
                    placeholder="Refine, e.g. 'add the parts of a white blood cell' or 'merge the vehicle groups'"
                    className="w-full px-4 py-2.5 pr-12 border border-ln2 rounded-xl text-sm focus:ring-2 focus:ring-ac focus:border-transparent placeholder:text-t3 disabled:bg-well"
                  />
                  <button
                    onClick={handleRefine}
                    disabled={generating || !refineText.trim()}
                    className="absolute top-1/2 -translate-y-1/2 right-2 flex items-center justify-center w-8 h-8 rounded-6 bg-accent text-onAccent hover:brightness-110 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Refine draft"
                  >
                    {generating ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Wand2 className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div className="flex justify-end gap-3 px-6 py-4 bg-well border-t border-ln">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-t2 bg-p1 border border-ln2 rounded-lg hover:bg-hv transition-colors"
            >
              {stage === "review" ? "Cancel" : "Close"}
            </button>
            {stage === "review" && (
              <button
                onClick={handleApply}
                disabled={!canApply}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-onAccent bg-accent rounded-lg hover:brightness-110 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {applying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                {applying ? "Applying…" : "Apply label space"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DescribeLabelSpaceModal;
