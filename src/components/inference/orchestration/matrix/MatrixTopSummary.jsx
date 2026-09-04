import React from "react";
import { ORCHESTRATION_CATEGORIES } from "../orchestrationViewModel";

/**
 * MatrixTopSummary Component (Design B)
 *
 * 3 category metric cards across the top:
 * - Interactive segmentation (Canvas prompts + within-image)
 * - Instance segmentation (Batch across the dataset)
 * - Cross-image suggestion (Retrieved exemplars)
 */
export default function MatrixTopSummary({ coverage }) {
  const categories = coverage?.categories || {};

  const getCategoryMeta = (catKey) => {
    switch (catKey) {
      case "interactive":
        return {
          subtitle: "Canvas prompts · Within-image",
          dotColor: "bg-teal-400 shadow-teal-400/50",
          barColor: "bg-teal-400",
          badgeBg: "bg-teal-500/10 text-teal-400 border-teal-500/20",
        };
      case "instance":
      case "batch-instance":
        return {
          subtitle: "Batch across whole dataset",
          dotColor: "bg-amber-400 shadow-amber-400/50",
          barColor: "bg-amber-400",
          badgeBg: "bg-amber-500/10 text-amber-400 border-amber-500/20",
        };
      case "cross-image":
        return {
          subtitle: "Exemplars across images",
          dotColor: "bg-purple-400 shadow-purple-400/50",
          barColor: "bg-purple-400",
          badgeBg: "bg-purple-500/10 text-purple-400 border-purple-500/20",
        };
      default:
        return {
          subtitle: "",
          dotColor: "bg-teal-400",
          barColor: "bg-teal-400",
          badgeBg: "bg-teal-500/10 text-teal-400 border-teal-500/20",
        };
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5" data-testid="matrix-top-summary">
      {ORCHESTRATION_CATEGORIES.map((cat) => {
        const catCov = categories[cat.key] || { bound: 0, possible: 0, percentage: 0 };
        const meta = getCategoryMeta(cat.key);
        const percentage = Math.min(100, Math.max(0, catCov.percentage || 0));

        return (
          <div
            key={cat.key}
            className="p-4 rounded-2xl border border-ln bg-p1 flex flex-col justify-between gap-3 shadow-xs hover:border-ln2 transition-all group"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full shrink-0 shadow-xs ${meta.dotColor}`} />
                  <h4 className="text-xs font-bold text-t1 tracking-tight">
                    {cat.label}
                  </h4>
                </div>
                <p className="text-[11px] text-t3 mt-0.5 pl-4">
                  {meta.subtitle}
                </p>
              </div>

              <div className="flex items-baseline gap-1 text-xs shrink-0 font-mono">
                <span className="text-sm font-bold text-t1">
                  {catCov.bound}
                </span>
                <span className="text-t3 text-[10px] font-sans">
                  / {catCov.possible} configured
                </span>
              </div>
            </div>

            {/* Progress track */}
            <div className="w-full bg-well rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${meta.barColor}`}
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
