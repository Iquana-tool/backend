import { describe, it, expect } from "vitest";
import {
  ORCHESTRATION_CATEGORIES,
  ROUTE_STATUS,
  normalizeSelector,
  parseSelectorKey,
  normalizeBinding,
  normalizeBindingsMap,
  findDuplicateSelectors,
  deepEqual,
  areInputsEqual,
  calculateCoverage,
  resolveEffectiveRoute,
  calculatePolicyDiff,
  formatChangeSummary,
} from "./orchestrationViewModel";
import { TASK_ORDER } from "../../../constants/tasks";

const makeModel = (task, registryKey, overrides = {}) => ({
  task,
  registry_key: registryKey,
  ...overrides,
});

const makeBinding = (task, labelId, modelRegistryKey, overrides = {}) => ({
  task,
  label_id: labelId,
  model_registry_key: modelRegistryKey,
  ...overrides,
});

const makeCatalog = (models) => ({ models });

describe("orchestrationViewModel", () => {
  describe("ORCHESTRATION_CATEGORIES mapping", () => {
    it("maps 3 visual categories to the 4 canonical backend task keys without losing any", () => {
      expect(ORCHESTRATION_CATEGORIES).toHaveLength(3);
      const allMappedTasks = ORCHESTRATION_CATEGORIES.flatMap((c) => c.tasks);
      expect(allMappedTasks).toHaveLength(4);
      expect(new Set(allMappedTasks)).toEqual(new Set(TASK_ORDER));

      const interactive = ORCHESTRATION_CATEGORIES.find((c) => c.key === "interactive");
      expect(interactive.tasks).toEqual(["prompted-segmentation", "instance-suggestion"]);

      const instance = ORCHESTRATION_CATEGORIES.find((c) => c.key === "instance");
      expect(instance.tasks).toEqual(["instance-segmentation"]);

      const crossImage = ORCHESTRATION_CATEGORIES.find((c) => c.key === "cross-image");
      expect(crossImage.tasks).toEqual(["cross-image-suggestion"]);
    });
  });

  describe("Selector normalization and parsing", () => {
    it("normalizes task default and label selector keys correctly", () => {
      expect(normalizeSelector("prompted-segmentation", null)).toBe("prompted-segmentation::default");
      expect(normalizeSelector("prompted-segmentation", undefined)).toBe("prompted-segmentation::default");
      expect(normalizeSelector("prompted-segmentation", "")).toBe("prompted-segmentation::default");
      expect(normalizeSelector("instance-segmentation", 42)).toBe("instance-segmentation::42");
      expect(normalizeSelector("instance-segmentation", "42")).toBe("instance-segmentation::42");
    });

    it("parses selector keys back to task and numeric labelId", () => {
      expect(parseSelectorKey("prompted-segmentation::default")).toEqual({
        task: "prompted-segmentation",
        labelId: null,
      });
      expect(parseSelectorKey("instance-segmentation::105")).toEqual({
        task: "instance-segmentation",
        labelId: 105,
      });
    });

    it("normalizes binding objects into canonical shapes", () => {
      const raw = {
        task: "cross-image-suggestion",
        label_id: "12",
        model_registry_key: "sam_model",
        inputs: {
          conditioning: { count: 10 },
          parameters: { threshold: 0.5 },
        },
      };
      const norm = normalizeBinding(raw);
      expect(norm).toEqual({
        task: "cross-image-suggestion",
        label_id: 12,
        model_registry_key: "sam_model",
        inputs: {
          conditioning: { count: 10 },
          parameters: { threshold: 0.5 },
        },
      });
    });

    it("detects duplicate selector bindings", () => {
      const bindings = [
        makeBinding("prompted-segmentation", null, "m1"),
        makeBinding("prompted-segmentation", null, "m2"),
        makeBinding("instance-segmentation", 1, "m3"),
        makeBinding("instance-segmentation", 1, "m4"),
        makeBinding("cross-image-suggestion", 2, "m5"),
      ];
      const dupes = findDuplicateSelectors(bindings);
      expect(dupes).toEqual(["prompted-segmentation::default", "instance-segmentation::1"]);
    });
  });

  describe("Structural equality of inputs", () => {
    it("handles key-order independence and deep equality", () => {
      const inA = {
        conditioning: { a: 1, b: 2 },
        parameters: { x: "hello", y: [1, 2, 3] },
      };
      const inB = {
        conditioning: { b: 2, a: 1 },
        parameters: { y: [1, 2, 3], x: "hello" },
      };
      const inC = {
        conditioning: { a: 1, b: 3 },
        parameters: { x: "hello", y: [1, 2, 3] },
      };
      expect(areInputsEqual(inA, inB)).toBe(true);
      expect(areInputsEqual(inA, inC)).toBe(false);
      expect(deepEqual(inA, inB)).toBe(true);
    });
  });

  describe("Coverage calculation", () => {
    const mockCatalog = makeCatalog([
      makeModel("prompted-segmentation", "prompted_1"),
      makeModel("instance-suggestion", "sugg_1"),
      makeModel("instance-segmentation", "inst_1"),
      makeModel("cross-image-suggestion", "cross_1"),
    ]);

    it("calculates 0-label denominator correctly (4 possible task defaults)", () => {
      const coverage = calculateCoverage([], {}, mockCatalog);
      expect(coverage.overall).toEqual({
        bound: 0,
        possible: 4,
        stale: 0,
        percentage: 0,
      });
      expect(coverage.categories.interactive.possible).toBe(2);
      expect(coverage.categories.instance.possible).toBe(1);
      expect(coverage.categories["cross-image"].possible).toBe(1);
    });

    it("calculates coverage with 3 labels correctly across all categories", () => {
      const labels = { 1: { id: 1 }, 2: { id: 2 }, 3: { id: 3 } };
      // slotCount = 1 + 3 = 4 slots per task.
      // Overall = 4 * 4 = 16.
      // Interactive = 4 * 2 = 8.
      // Instance = 4 * 1 = 4.
      // Cross-image = 4 * 1 = 4.

      const draftBindings = [
        makeBinding("prompted-segmentation", null, "prompted_1"),
        makeBinding("prompted-segmentation", 1, "prompted_1"),
        makeBinding("instance-suggestion", null, "sugg_1"),
        makeBinding("instance-segmentation", null, "stale_key"), // stale!
      ];

      const coverage = calculateCoverage(draftBindings, labels, mockCatalog);

      expect(coverage.overall.possible).toBe(16);
      expect(coverage.overall.bound).toBe(4);
      expect(coverage.overall.stale).toBe(1);
      expect(coverage.overall.percentage).toBe(25);

      expect(coverage.categories.interactive).toEqual({
        bound: 3,
        possible: 8,
        stale: 0,
        percentage: 38,
      });

      expect(coverage.categories.instance).toEqual({
        bound: 1,
        possible: 4,
        stale: 1,
        percentage: 25,
      });

      expect(coverage.categories["cross-image"]).toEqual({
        bound: 0,
        possible: 4,
        stale: 0,
        percentage: 0,
      });
    });

    it("reports 100% when all selectors are bound", () => {
      const labels = { 1: { id: 1 } }; // slotCount = 2 per task -> 8 total
      const allBindings = [
        makeBinding("prompted-segmentation", null, "prompted_1"),
        makeBinding("prompted-segmentation", 1, "prompted_1"),
        makeBinding("instance-suggestion", null, "sugg_1"),
        makeBinding("instance-suggestion", 1, "sugg_1"),
        makeBinding("instance-segmentation", null, "inst_1"),
        makeBinding("instance-segmentation", 1, "inst_1"),
        makeBinding("cross-image-suggestion", null, "cross_1"),
        makeBinding("cross-image-suggestion", 1, "cross_1"),
      ];
      const coverage = calculateCoverage(allBindings, labels, mockCatalog);
      expect(coverage.overall.bound).toBe(8);
      expect(coverage.overall.possible).toBe(8);
      expect(coverage.overall.percentage).toBe(100);
      expect(coverage.overall.stale).toBe(0);
    });

    it("ignores bindings for deleted labels and does not exceed 100% coverage", () => {
      const activeLabels = { 1: { id: 1 } }; // 1 active label -> slotCount = 2 (default + 1)
      const bindingsWithDeletedLabels = [
        makeBinding("instance-segmentation", null, "inst_1"),
        makeBinding("instance-segmentation", 1, "inst_1"),
        // Deleted labels (99, 100) that no longer exist in activeLabels
        makeBinding("instance-segmentation", 99, "inst_1"),
        makeBinding("instance-segmentation", 100, "inst_1"),
      ];

      const coverage = calculateCoverage(bindingsWithDeletedLabels, activeLabels, mockCatalog);
      // Instance category should only count default + label 1 (bound = 2, possible = 2, percentage = 100%)
      expect(coverage.categories.instance.bound).toBe(2);
      expect(coverage.categories.instance.possible).toBe(2);
      expect(coverage.categories.instance.percentage).toBe(100);
    });
  });

  describe("Effective route resolution", () => {
    const catalog = makeCatalog([
      makeModel("instance-segmentation", "general_seg", {
        name: "General Segmenter",
        label_ids: [], // class-agnostic
      }),
      makeModel("instance-segmentation", "specialist_seg", {
        name: "Cell Specialist",
        label_ids: [10, 20], // class-specific
      }),
      makeModel("instance-segmentation", "label30_seg", {
        name: "Organ Specialist",
        label_ids: [30],
      }),
    ]);

    it("resolves explicit healthy override", () => {
      const bindings = [
        makeBinding("instance-segmentation", 30, "label30_seg"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 30,
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.EXPLICIT);
      expect(res.model?.name).toBe("Organ Specialist");
      expect(res.isExplicit).toBe(true);
    });

    it("resolves explicit stale override when model is not in catalog", () => {
      const bindings = [
        makeBinding("instance-segmentation", 30, "deleted_model"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 30,
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.STALE);
      expect(res.model).toBeNull();
      expect(res.isExplicit).toBe(true);
    });

    it("resolves explicit stale override when model exists but does not cover the label", () => {
      const bindings = [
        // specialist_seg only covers [10, 20], so binding it to label 30 is incompatible
        makeBinding("instance-segmentation", 30, "specialist_seg"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 30,
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.STALE);
      expect(res.reason).toBe("incompatible_model");
      expect(res.isExplicit).toBe(true);
    });

    it("resolves unbound when neither default nor override exists", () => {
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 10,
        bindings: [],
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.UNBOUND);
      expect(res.reason).toBe("no_default");
    });

    it("resolves inherited route for class-agnostic task default", () => {
      const bindings = [
        makeBinding("instance-segmentation", null, "general_seg"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 99,
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.INHERITED);
      expect(res.model?.registry_key).toBe("general_seg");
      expect(res.inheritedFrom).toBe("task-default");
    });

    it("resolves inherited route for class-specific default covering the label", () => {
      const bindings = [
        makeBinding("instance-segmentation", null, "specialist_seg"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 10,
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.INHERITED);
      expect(res.model?.name).toBe("Cell Specialist");
    });

    it("resolves unbound-incompatible-default when class-specific default excludes the label", () => {
      const bindings = [
        makeBinding("instance-segmentation", null, "specialist_seg"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 99, // not in [10, 20]
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.UNBOUND_INCOMPATIBLE);
      expect(res.reason).toBe("incompatible_default");
    });

    it("resolves stale-default when task default model is missing in catalog", () => {
      const bindings = [
        makeBinding("instance-segmentation", null, "unregistered_key"),
      ];
      const res = resolveEffectiveRoute({
        task: "instance-segmentation",
        labelId: 10,
        bindings,
        catalog,
      });
      expect(res.status).toBe(ROUTE_STATUS.STALE_DEFAULT);
      expect(res.reason).toBe("stale_default");
    });
  });

  describe("calculatePolicyDiff and formatChangeSummary", () => {
    const catalog = makeCatalog([
      makeModel("prompted-segmentation", "sam_v1", { name: "SAM Base" }),
      makeModel("prompted-segmentation", "sam_v2", { name: "SAM Large" }),
    ]);
    const labelsById = {
      1: { id: 1, name: "Nuclei" },
      2: { id: 2, name: "Cytoplasm" },
    };

    it("returns empty diff when saved and draft are identical", () => {
      const saved = [
        makeBinding("prompted-segmentation", null, "sam_v1", {
          inputs: { parameters: { iou_thresh: 0.8 } },
        }),
      ];
      const draft = [
        makeBinding("prompted-segmentation", null, "sam_v1", {
          inputs: { parameters: { iou_thresh: 0.8 } },
        }),
      ];
      const diff = calculatePolicyDiff(saved, draft);
      expect(diff).toEqual([]);

      const summary = formatChangeSummary(diff, { labelsById, catalog });
      expect(summary.totalCount).toBe(0);
      expect(summary.summaryText).toBe("No unsaved changes");
    });

    it("detects added, removed, model_changed, and inputs_changed", () => {
      const saved = [
        makeBinding("prompted-segmentation", null, "sam_v1", { inputs: {} }),
        makeBinding("prompted-segmentation", 1, "sam_v1", { inputs: { parameters: { th: 0.5 } } }),
        makeBinding("prompted-segmentation", 2, "sam_v1", { inputs: {} }),
      ];

      const draft = [
        // Changed model for default
        makeBinding("prompted-segmentation", null, "sam_v2", { inputs: {} }),
        // Changed inputs for label 1
        makeBinding("prompted-segmentation", 1, "sam_v1", { inputs: { parameters: { th: 0.9 } } }),
        // Removed label 2
        // Added cross-image default
        makeBinding("cross-image-suggestion", null, "cross_m", { inputs: {} }),
      ];

      const diff = calculatePolicyDiff(saved, draft);
      expect(diff).toHaveLength(4);

      const types = diff.map((d) => d.type);
      expect(types).toContain("model_changed");
      expect(types).toContain("inputs_changed");
      expect(types).toContain("added");
      expect(types).toContain("removed");

      const summary = formatChangeSummary(diff, { labelsById, catalog });
      expect(summary.totalCount).toBe(4);
      expect(summary.items).toHaveLength(4);
      expect(summary.summaryText).toContain("(+2 more)");
    });
  });
});
