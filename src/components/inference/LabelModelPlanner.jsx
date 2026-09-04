import React from "react";
import { ChevronRight, Cpu, Sparkles, Star, Layers } from "lucide-react";
import DynamicHyperParameter from "../datasets/training/DynamicHyperParameter";
import { BATCH_INFERENCE_TASKS } from "../../constants/tasks";
import {
    getEffectiveContract,
    initStep,
    updateStepConditioning,
    updateStepParameter,
} from "./plannerContractUtils";

/**
 * The orchestration editor: the dataset's label hierarchy, with a model bound to each label.
 *
 * This is the whole point of the page. A user who trained one specialist per class points
 * each label at its own model; a user with one multiclass model points several labels at it
 * and the backend filters its output down to each label in turn. Labels left on "Skip" are
 * simply not part of the run.
 *
 * The tree is rendered *by level*, not as a nested outline, because the level is what the run
 * actually does: everything at level 1 is annotated across the whole dataset before anything
 * at level 2 starts, so a child model always has parent instances to nest its predictions in.
 * The heading on each level block says so.
 */

const SKIP = "";

/** Group labels by hierarchy depth, with each label's parent name for the caption. */
export const groupLabelsByLevel = (labelsById) => {
    const labels = Object.values(labelsById || {});
    const depthOf = (label) => {
        let depth = 0;
        let current = label;
        // Hierarchies are a handful of levels deep; walking up is cheaper than a pre-pass.
        while (current?.parent_id != null && labelsById[current.parent_id]) {
            depth += 1;
            current = labelsById[current.parent_id];
        }
        return depth;
    };

    const byLevel = new Map();
    labels.forEach((label) => {
        const level = depthOf(label);
        if (!byLevel.has(level)) byLevel.set(level, []);
        byLevel.get(level).push({
            ...label,
            level,
            parentName: label.parent_id != null ? labelsById[label.parent_id]?.name : null,
        });
    });
    return [...byLevel.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([level, entries]) => ({
            level,
            labels: entries.sort((a, b) => a.name.localeCompare(b.name)),
        }));
};

/** Models that may be bound to a label: class-agnostic ones, plus those predicting it, filtered to allowed tasks. */
export const modelsForLabel = (models, labelId, allowedTasks = BATCH_INFERENCE_TASKS) =>
    (models || []).filter(
        (model) =>
            (!allowedTasks || allowedTasks.includes(model.task)) &&
            (!model.label_ids || model.label_ids.length === 0 || model.label_ids.includes(labelId))
    );

function ModelRow({ label, step, models, strategies, onChange }) {
    const options = modelsForLabel(models, label.id);
    const selected = models.find(
        (model) => step && model.registry_key === step.model_registry_key && model.task === step.task
    );
    const isLegacyFallback = selected?.provenance === "legacy_default";
    const contract = selected ? getEffectiveContract(selected) : null;
    const condSpec = contract?.conditioning;
    const usesRetrievalStrategy =
        condSpec?.kind === "reference_images" ||
        condSpec?.kind === "embeddings" ||
        (condSpec?.kind === "instances" &&
            (step?.inputs?.conditioning?.strategy != null || step?.retrieval_strategy != null));

    const setModel = (value) => {
        if (value === SKIP) return onChange(label.id, null);
        const [task, registryKey] = value.split("::");
        const found = models.find((m) => m.task === task && m.registry_key === registryKey);
        if (!found) return onChange(label.id, null);
        const newStep = initStep(label, found, strategies);
        onChange(label.id, newStep);
    };

    const handleParamChange = (key, val) => {
        const nextStep = updateStepParameter(step, key, val);
        onChange(label.id, nextStep);
    };

    const handleCondChange = (key, val) => {
        const nextStep = updateStepConditioning(step, key, val);
        onChange(label.id, nextStep);
    };

    const countUnitLabel =
        condSpec?.unit === "image"
            ? "Images"
            : condSpec?.unit === "instance"
            ? "Instances"
            : condSpec?.unit === "vector"
            ? "Vectors"
            : "Count";

    return (
        <div className="border border-ln rounded-xl bg-p1 overflow-hidden">
            <div className="flex flex-wrap items-center gap-3 px-3 py-2.5">
                <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                        {label.parentName && <ChevronRight size={13} className="text-t3 shrink-0" />}
                        <span className="text-sm font-medium text-t1 truncate">{label.name}</span>
                    </div>
                    {label.parentName && (
                        <p className="text-[11px] text-t3 mt-0.5">
                            nested inside {label.parentName}
                        </p>
                    )}
                </div>

                <select
                    value={step ? `${step.task}::${step.model_registry_key}` : SKIP}
                    onChange={(event) => setModel(event.target.value)}
                    className="w-full sm:w-72 px-3 py-1.5 text-sm border border-ln2 rounded-lg bg-well text-t1 focus:ring-2 focus:ring-ac focus:border-transparent"
                    aria-label={`Model for ${label.name}`}
                >
                    <option value={SKIP} className="bg-p1 text-t1">Skip this label</option>
                    {options.map((model) => (
                        <option
                            key={`${model.task}::${model.registry_key}`}
                            value={`${model.task}::${model.registry_key}`}
                            className="bg-p1 text-t1"
                        >
                            {model.name}
                            {model.trained_on_dataset ? " ★" : ""}
                            {model.task === "cross-image-suggestion" ? " (in-context)" : ""}
                        </option>
                    ))}
                    {options.length === 0 && <option disabled className="bg-p1 text-t3">No compatible model</option>}
                </select>
            </div>

            {step && selected && contract && (
                <div className="px-3 pb-3 pt-0.5 space-y-2.5 border-t border-ln bg-p2/30">
                    {/* Meta / Capabilities banner */}
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
                        {selected.trained_on_dataset && (
                            <span className="inline-flex items-center gap-1 text-ok font-medium">
                                <Star size={11} /> Trained on this dataset
                            </span>
                        )}
                        {selected.label_ids?.length > 1 ? (
                            <span className="text-t3">
                                Predicts {selected.label_ids.length} classes — output filtered to “{label.name}”.
                            </span>
                        ) : (!selected.label_ids || selected.label_ids.length === 0) ? (
                            <span className="text-t3">
                                Class-agnostic — everything it finds is labelled “{label.name}”.
                            </span>
                        ) : null}
                    </div>

                    {/* Legacy models keep this gateway-side post-filter outside the model contract. */}
                    {isLegacyFallback && (
                        <label
                            htmlFor={`label-${label.id}-min-confidence`}
                            className="inline-flex items-center gap-1.5 text-[11px] text-t2"
                        >
                            Min. confidence
                            <input
                                id={`label-${label.id}-min-confidence`}
                                type="number"
                                min={0}
                                max={1}
                                step={0.05}
                                value={step.min_confidence ?? 0}
                                onChange={(event) =>
                                    onChange(label.id, {
                                        ...step,
                                        min_confidence: Number(event.target.value),
                                    })
                                }
                                className="w-16 px-1.5 py-0.5 text-[11px] border border-ln rounded bg-well text-t1"
                                aria-label={`Min. confidence for ${label.name}`}
                            />
                        </label>
                    )}

                    {/* Dedicated Conditioning Section (when kind !== 'none') */}
                    {condSpec && condSpec.kind !== "none" && (
                        <div className="p-2.5 rounded-lg bg-well/60 border border-ln space-y-2">
                            <div className="flex items-center gap-1.5 text-[11px] font-semibold text-ac uppercase tracking-wider">
                                <Sparkles size={12} className="text-ac" />
                                <span>Conditioning &amp; Exemplars ({condSpec.kind === "reference_images" ? "Reference Images" : condSpec.kind === "instances" ? "Instances" : condSpec.kind === "concept_text" ? "Text Prompt" : condSpec.kind})</span>
                            </div>

                            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                                {/* Strategy for retrieval-backed conditioning */}
                                {usesRetrievalStrategy && (
                                    <label
                                        htmlFor={`label-${label.id}-strategy`}
                                        className="inline-flex items-center gap-1.5 text-xs text-t2"
                                    >
                                        <span className="font-medium text-t1">Exemplars:</span>
                                        <select
                                            id={`label-${label.id}-strategy`}
                                            value={step.inputs?.conditioning?.strategy || step.retrieval_strategy || ""}
                                            onChange={(event) => handleCondChange("strategy", event.target.value)}
                                            className="px-2 py-1 text-xs border border-ln2 rounded bg-well text-t1 focus:ring-1 focus:ring-ac"
                                            aria-label={`Retrieval strategy for ${label.name}`}
                                        >
                                            {strategies
                                                .filter((strategy) => strategy.available)
                                                .map((strategy) => (
                                                    <option key={strategy.key} value={strategy.key} className="bg-p1 text-t1">
                                                        {strategy.label || strategy.key}
                                                    </option>
                                                ))}
                                        </select>
                                    </label>
                                )}

                                {/* Count when user selectable */}
                                {condSpec.user_selectable_count ? (
                                    <label
                                        htmlFor={`label-${label.id}-count`}
                                        className="inline-flex items-center gap-1.5 text-xs text-t2"
                                    >
                                        <span className="font-medium text-t1">{countUnitLabel}:</span>
                                        <input
                                            id={`label-${label.id}-count`}
                                            type="number"
                                            min={condSpec.min_units ?? 1}
                                            max={condSpec.max_units ?? undefined}
                                            value={step.inputs?.conditioning?.count ?? step.top_k ?? condSpec.min_units ?? 1}
                                            onChange={(event) => handleCondChange("count", Number(event.target.value))}
                                            className="w-16 px-2 py-1 text-xs border border-ln2 rounded bg-well text-t1 focus:ring-1 focus:ring-ac"
                                            aria-label={`${countUnitLabel} for ${label.name}`}
                                        />
                                    </label>
                                ) : (condSpec.kind === "reference_images" || condSpec.kind === "instances") ? (
                                    <span className="text-[11px] text-t3 bg-p1 px-2 py-0.5 rounded border border-ln">
                                        {(() => {
                                            const count = step.inputs?.conditioning?.count ?? condSpec.max_units ?? condSpec.min_units ?? 1;
                                            const unit = condSpec.unit || (condSpec.kind === "reference_images" ? "reference image" : "instance");
                                            const unitStr = count === 1 ? unit : `${unit}s`;
                                            return `${count} ${unitStr} (model fixed)`;
                                        })()}
                                    </span>
                                ) : null}

                                {/* Concept text */}
                                {condSpec.kind === "concept_text" && (
                                    <label
                                        htmlFor={`label-${label.id}-concept-text`}
                                        className="inline-flex items-center gap-1.5 text-xs text-t2 flex-1 min-w-[200px]"
                                    >
                                        <span className="font-medium text-t1">Prompt:</span>
                                        <input
                                            id={`label-${label.id}-concept-text`}
                                            type="text"
                                            value={step.inputs?.conditioning?.concept_text ?? label.name}
                                            onChange={(e) => handleCondChange("concept_text", e.target.value)}
                                            className="flex-1 px-2 py-1 text-xs border border-ln2 rounded bg-well text-t1 focus:ring-1 focus:ring-ac"
                                            aria-label={`Prompt for ${label.name}`}
                                        />
                                    </label>
                                )}
                            </div>

                            {/* Contract notes */}
                            {contract.notes && (
                                <p className="text-[11px] text-t3 pt-0.5 leading-relaxed">{contract.notes}</p>
                            )}
                        </div>
                    )}

                    {/* Model Parameters Section */}
                    {contract.parameters && contract.parameters.length > 0 && (
                        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pt-0.5">
                            {contract.parameters.map((param) => (
                                <DynamicHyperParameter
                                    key={param.key}
                                    param={param}
                                    value={step.inputs?.parameters?.[param.key]}
                                    onChange={handleParamChange}
                                    compact
                                    idPrefix={`label-${label.id}`}
                                />
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default function LabelModelPlanner({
    labelsById,
    models,
    strategies,
    steps,
    onChange,
    allowedTasks = BATCH_INFERENCE_TASKS,
}) {
    const levels = groupLabelsByLevel(labelsById);
    const stepByLabel = new Map(steps.map((step) => [step.label_id, step]));
    const batchModels = React.useMemo(() => {
        return (models || []).filter((m) => !allowedTasks || allowedTasks.includes(m.task));
    }, [models, allowedTasks]);

    if (levels.length === 0) {
        return (
            <p className="text-sm text-t3 p-4 border border-dashed border-ln2 rounded-xl">
                This dataset has no labels yet. Create the label hierarchy first.
            </p>
        );
    }

    return (
        <div className="space-y-5">
            {levels.map(({ level, labels }, index) => (
                <section key={level}>
                    <div className="flex items-center gap-2 mb-2">
                        <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-ac bg-acS px-2 py-0.5 rounded-full">
                            <Layers size={11} /> Level {level + 1}
                        </span>
                        <p className="text-[11px] text-t3">
                            {index === 0
                                ? "Runs first, across every image in scope."
                                : "Runs once the level above has finished the whole dataset, so predictions can be nested inside their parents."}
                        </p>
                    </div>
                    <div className="space-y-2">
                        {labels.map((label) => (
                            <ModelRow
                                key={label.id}
                                label={label}
                                step={stepByLabel.get(label.id) || null}
                                models={batchModels}
                                strategies={strategies}
                                onChange={onChange}
                            />
                        ))}
                    </div>
                </section>
            ))}

            <p className="flex items-start gap-2 text-[11px] text-t3">
                <Cpu size={13} className="shrink-0 mt-0.5" />
                Models marked ★ were trained on this dataset. A model that predicts several
                classes can be bound to more than one label — its output is filtered down to
                whichever label it is bound to, so mixing specialists and multiclass models in
                one run is fine.
            </p>
            {batchModels.some(
                (model) =>
                    model.input_contract?.conditioning?.kind === "reference_images" ||
                    model.input_contract?.conditioning?.kind === "instances" ||
                    model.input_contract?.conditioning?.kind === "embeddings"
            ) && (
                <p className="flex items-start gap-2 text-[11px] text-t3">
                    <Sparkles size={13} className="shrink-0 mt-0.5" />
                    In-context models annotate by example: they pull exemplars of the label from
                    other images in the dataset instead of relying solely on fixed weights.
                </p>
            )}
        </div>
    );
}
