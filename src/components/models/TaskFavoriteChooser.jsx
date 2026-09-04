import React, { useEffect, useRef } from "react";
import { Star } from "lucide-react";
import { getTaskMeta } from "../../constants/tasks";

/**
 * Small popover task chooser when starring a multi-task model from the 'All' view.
 * Allows setting or removing personal task defaults per task independently.
 */
export default function TaskFavoriteChooser({
  model,
  favorites = {},
  onToggleTaskFavorite,
  onClose,
  anchorAlign = "right",
}) {
  const ref = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (ref.current && !ref.current.contains(e.target)) {
        onClose?.();
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  const tasks = model?.tasks || [];

  return (
    <div
      ref={ref}
      className={`absolute z-30 top-full mt-1.5 ${
        anchorAlign === "right" ? "right-0" : "left-0"
      } w-56 rounded-xl bg-p1 border border-ln shadow-lg p-2 text-left animate-in fade-in zoom-in-95 duration-100`}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="text-[11px] font-semibold text-t3 px-2 py-1 uppercase tracking-wider">
        Star default for:
      </div>
      <div className="space-y-1 mt-1">
        {tasks.map((taskKey) => {
          const meta = getTaskMeta(taskKey);
          const isFav = favorites[taskKey] === model.identifier;
          return (
            <button
              key={taskKey}
              type="button"
              onClick={() => onToggleTaskFavorite?.(model, taskKey)}
              aria-label={`Toggle favorite for ${meta.label}`}
              className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-xs text-t1 hover:bg-hv transition-colors"
            >
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${meta.dot}`} />
                <span>{meta.short || meta.label}</span>
              </div>
              <Star
                className={`w-3.5 h-3.5 ${
                  isFav ? "fill-amber-400 text-warn" : "text-t3 hover:text-warn"
                }`}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}
