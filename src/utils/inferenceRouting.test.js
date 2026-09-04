import { describe, it, expect } from "vitest";
import {
    isModelCompatibleWithLabel,
    resolveRoutingBinding,
    getBatchStepsFromPolicy,
} from "./inferenceRouting";

const makeModel = (task, registryKey, labelIds) => ({
    registry_key: registryKey,
    task,
    label_ids: labelIds,
});

const makeBinding = (task, labelId, modelRegistryKey, overrides = {}) => ({
    task,
    label_id: labelId,
    model_registry_key: modelRegistryKey,
    ...overrides,
});

const makeCatalog = (...models) => models;

const makePolicy = (bindings, overrides = {}) => ({
    dataset_id: 10,
    bindings,
    ...overrides,
});

describe("inferenceRouting utilities", () => {
    const catalogModels = makeCatalog(
        makeModel("instance-segmentation", "m2f-generic", []),
        makeModel("instance-segmentation", "m2f-cell-specialist", [1]),
        makeModel("cross-image-suggestion", "sam3-cross", []),
        makeModel("cross-image-suggestion", "sam3-cross-nucleus", [2]),
    );

    const labelsById = {
        1: { id: 1, name: "cell" },
        2: { id: 2, name: "nucleus" },
        3: { id: 3, name: "background" },
    };

    it("evaluates model label compatibility accurately", () => {
        const classAgnostic = { label_ids: [] };
        const specialist = { label_ids: [1, 2] };

        expect(isModelCompatibleWithLabel(classAgnostic, 1)).toBe(true);
        expect(isModelCompatibleWithLabel(classAgnostic, 99)).toBe(true);
        expect(isModelCompatibleWithLabel(specialist, 1)).toBe(true);
        expect(isModelCompatibleWithLabel(specialist, 2)).toBe(true);
        expect(isModelCompatibleWithLabel(specialist, 3)).toBe(false);
    });

    it("prefers exact label override over task default", () => {
        const policy = makePolicy([
            makeBinding("instance-segmentation", null, "m2f-generic"),
            makeBinding("instance-segmentation", 1, "m2f-cell-specialist"),
        ]);

        const resolvedCell = resolveRoutingBinding(
            policy,
            "instance-segmentation",
            1,
            catalogModels
        );
        expect(resolvedCell.isOverride).toBe(true);
        expect(resolvedCell.binding.model_registry_key).toBe("m2f-cell-specialist");
        expect(resolvedCell.isCompatible).toBe(true);

        const resolvedNucleus = resolveRoutingBinding(
            policy,
            "instance-segmentation",
            2,
            catalogModels
        );
        expect(resolvedNucleus.isOverride).toBe(false);
        expect(resolvedNucleus.isTaskDefault).toBe(true);
        expect(resolvedNucleus.binding.model_registry_key).toBe("m2f-generic");
        expect(resolvedNucleus.isCompatible).toBe(true);
    });

    it("safely handles class-aware task defaults", () => {
        const policy = makePolicy([
            makeBinding("cross-image-suggestion", null, "sam3-cross-nucleus"), // only predicts label 2
        ]);

        // Label 2 (nucleus): compatible
        const resolvedNucleus = resolveRoutingBinding(
            policy,
            "cross-image-suggestion",
            2,
            catalogModels
        );
        expect(resolvedNucleus.isTaskDefault).toBe(true);
        expect(resolvedNucleus.isCompatible).toBe(true);

        // Label 1 (cell): NOT compatible with the class-aware default
        const resolvedCell = resolveRoutingBinding(
            policy,
            "cross-image-suggestion",
            1,
            catalogModels
        );
        expect(resolvedCell.isTaskDefault).toBe(true);
        expect(resolvedCell.isCompatible).toBe(false);
    });

    it("flags stale bindings when the model is not in the live catalog", () => {
        const policy = makePolicy([
            makeBinding("instance-segmentation", 1, "retired-model"),
        ]);

        const resolved = resolveRoutingBinding(
            policy,
            "instance-segmentation",
            1,
            catalogModels
        );
        expect(resolved.isStale).toBe(true);
        expect(resolved.isCompatible).toBe(false);
        expect(resolved.model).toBeNull();

        // Stale model is never included in executable batch steps
        const steps = getBatchStepsFromPolicy(policy, labelsById, catalogModels);
        expect(steps[1]).toBeUndefined();
    });

    it("marks models as stale when catalog is successfully loaded as empty array", () => {
        const policy = makePolicy([
            makeBinding("cross-image-suggestion", 1, "sam3-cross"),
        ]);

        // When catalog loaded with 0 models
        const resolvedEmpty = resolveRoutingBinding(
            policy,
            "cross-image-suggestion",
            1,
            []
        );
        expect(resolvedEmpty.isStale).toBe(true);
        expect(resolvedEmpty.isCompatible).toBe(false);

        // When catalog was not provided (null)
        const resolvedUnprovided = resolveRoutingBinding(
            policy,
            "cross-image-suggestion",
            1,
            null
        );
        expect(resolvedUnprovided.isStale).toBe(false);
        expect(resolvedUnprovided.isCompatible).toBe(true);
    });

    it("constructs batch steps deterministically with preferredTask filter", () => {
        const policy = makePolicy([
            makeBinding("instance-segmentation", null, "m2f-generic"),
            makeBinding("cross-image-suggestion", 1, "sam3-cross"),
        ]);

        // Default prioritization: instance-segmentation default covers 1, 2, 3
        const allSteps = getBatchStepsFromPolicy(policy, labelsById, catalogModels);
        expect(allSteps[1].model_registry_key).toBe("m2f-generic");
        expect(allSteps[2].model_registry_key).toBe("m2f-generic");

        // Explicit cross-image preference: label 1 gets sam3-cross, label 2 has no cross-image default/override
        const crossSteps = getBatchStepsFromPolicy(
            policy,
            labelsById,
            catalogModels,
            "cross-image-suggestion"
        );
        expect(crossSteps[1].model_registry_key).toBe("sam3-cross");
        expect(crossSteps[2]).toBeUndefined();
    });
});
