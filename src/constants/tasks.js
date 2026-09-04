// Shared task metadata for the model-centric zoo, cards, and filters.
//
// A model can serve several tasks (e.g. SAM 3 does prompted segmentation AND
// instance suggestion), so tasks are a first-class facet rather than a single
// "service" a model belongs to. `key` matches the ai-service task surface /
// backend task tag; `endpoint` is the gateway route the models are listed under.
//
// Tailwind class strings are written out in full (not interpolated) so the
// compiler keeps them.
export const TASKS = {
  "prompted-segmentation": {
    key: "prompted-segmentation",
    endpoint: "prompted_segmentation",
    short: "Prompted",
    label: "Prompted segmentation",
    description: "Interactive point-and-box segmentation for guided annotation.",
    chip: "bg-acS text-ac",
    dot: "bg-accent",
  },
  "instance-suggestion": {
    key: "instance-suggestion",
    endpoint: "suggestion_segmentation",
    short: "Within-Image",
    label: "Within-image suggestion",
    description: "Auto-detect instances within the current image.",
    chip: "bg-acS text-ac",
    dot: "bg-accent",
  },
  "instance-segmentation": {
    key: "instance-segmentation",
    endpoint: "instance_segmentation",
    short: "Full-Image",
    label: "Instance segmentation",
    description: "Batch instance segmentation across dataset images.",
    chip: "bg-warnBg text-warn",
    dot: "bg-warn",
  },
  "cross-image-suggestion": {
    key: "cross-image-suggestion",
    endpoint: "cross_image_suggestion",
    short: "Cross-Image",
    label: "Cross-image suggestion",
    description: "Suggest instances by retrieving exemplars across dataset images.",
    chip: "bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300",
    dot: "bg-purple-500",
  },
};

// Canonical display and policy order across the application
export const TASK_ORDER = [
  "prompted-segmentation",
  "instance-suggestion",
  "instance-segmentation",
  "cross-image-suggestion",
];

// Tasks supported for full-dataset batch execution
export const BATCH_INFERENCE_TASKS = [
  "instance-segmentation",
  "cross-image-suggestion",
];

// Tasks supported for single-image interactive suggestion
export const INTERACTIVE_SUGGESTION_TASKS = [
  "cross-image-suggestion",
];

export const getTaskMeta = (key) => TASKS[key] || {
  key,
  short: key,
  label: key,
  description: "",
  chip: "bg-well text-t2",
  dot: "bg-t3",
};
