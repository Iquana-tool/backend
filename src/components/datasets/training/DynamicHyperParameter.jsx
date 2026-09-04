import React from "react";
import { Info } from "lucide-react";

function ParameterHelp({ label, description, tooltipId }) {
  if (!description) return null;

  return (
    <span className="relative inline-flex items-center group">
      <span
        role="img"
        tabIndex={0}
        aria-label={`About ${label}: ${description}`}
        aria-describedby={tooltipId}
        className="inline-flex h-3.5 w-3.5 items-center justify-center text-t3 hover:text-ac focus:outline-none focus-visible:ring-1 focus-visible:ring-ac"
      >
        <Info size={12} strokeWidth={2.25} aria-hidden="true" />
      </span>
      <span
        id={tooltipId}
        role="tooltip"
        className="pointer-events-none invisible absolute bottom-full left-0 z-50 mb-1 w-56 rounded-md border border-ln2 bg-p1 px-2 py-1.5 text-[11px] font-normal leading-snug text-t1 opacity-0 shadow-lg group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
      >
        {description}
      </span>
    </span>
  );
}

/**
 * Renders a single parameter input, driven by the model-declared descriptor
 * (see iquana_toolbox HyperParameter). The widget is inferred from the descriptor:
 *   - `options` set            -> dropdown
 *   - type "bool"              -> checkbox
 *   - `min_value`/`max_value`  -> slider (for int/float with finite bounds)
 *   - type "int"/"float"       -> number input
 *   - otherwise                -> text input (type "str" or unconstrained)
 */
export const coerceValue = (raw, type) => {
  if (type === "int") {
    if (raw === "" || raw === null || raw === undefined) return "";
    const val = parseInt(raw, 10);
    return Number.isNaN(val) ? "" : val;
  }
  if (type === "float") {
    if (raw === "" || raw === null || raw === undefined) return "";
    const val = parseFloat(raw);
    return Number.isNaN(val) ? "" : val;
  }
  if (type === "bool") {
    if (typeof raw === "boolean") return raw;
    if (raw === "true") return true;
    if (raw === "false") return false;
    return Boolean(raw);
  }
  if (type === "str") return String(raw ?? "");
  return raw;
};

export default function DynamicHyperParameter({
  param,
  value,
  onChange,
  compact = false,
  idPrefix = "",
  disabled = false,
}) {
  const { key, label, description, type, options, min_value, max_value, step } = param;
  const displayLabel = label || key;
  const current = value ?? param.default_value;
  const inputId = idPrefix ? `${idPrefix}-param-${key}` : `param-${key}`;
  const descriptionId = `${inputId}-description`;

  const isSlider = min_value !== null && min_value !== undefined &&
    max_value !== null && max_value !== undefined && !options &&
    (type === "int" || type === "float");

  let control;
  if (options && options.length > 0) {
    control = (
      <select
        id={inputId}
        aria-label={label || key}
        disabled={disabled}
        value={current !== undefined && current !== null ? String(current) : ""}
        onChange={(e) => onChange(key, coerceValue(e.target.value, type))}
        className="w-full px-2.5 py-1.5 text-xs sm:text-sm border border-ln2 rounded-lg bg-well text-t1 focus:ring-2 focus:ring-ac focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {options.map((opt) => (
          <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
        ))}
      </select>
    );
  } else if (type === "bool") {
    control = (
      <input
        id={inputId}
        aria-label={label || key}
        type="checkbox"
        disabled={disabled}
        checked={Boolean(current)}
        onChange={(e) => onChange(key, e.target.checked)}
        className="h-4 w-4 rounded border-ln2 text-ac focus:ring-ac disabled:opacity-50 disabled:cursor-not-allowed"
      />
    );
  } else if (isSlider) {
    control = (
      <div className="flex items-center gap-2">
        <input
          id={inputId}
          aria-label={label || key}
          type="range"
          disabled={disabled}
          min={min_value}
          max={max_value}
          step={step ?? (type === "int" ? 1 : 0.01)}
          value={current ?? min_value}
          onChange={(e) => onChange(key, coerceValue(e.target.value, type))}
          className="flex-1 accent-ac disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <span className="text-xs font-mono font-semibold w-12 text-right text-t1">{current}</span>
      </div>
    );
  } else if (type === "int" || type === "float") {
    control = (
      <input
        id={inputId}
        aria-label={label || key}
        type="number"
        disabled={disabled}
        min={min_value !== null && min_value !== undefined ? min_value : undefined}
        max={max_value !== null && max_value !== undefined ? max_value : undefined}
        step={step ?? (type === "int" ? 1 : "any")}
        value={current !== undefined && current !== null ? current : ""}
        onChange={(e) => onChange(key, coerceValue(e.target.value, type))}
        className="w-full px-2.5 py-1.5 text-xs sm:text-sm border border-ln2 rounded-lg bg-well text-t1 focus:ring-2 focus:ring-ac focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
      />
    );
  } else {
    control = (
      <input
        id={inputId}
        aria-label={label || key}
        type="text"
        disabled={disabled}
        value={current !== undefined && current !== null ? String(current) : ""}
        onChange={(e) => onChange(key, coerceValue(e.target.value, type))}
        className="w-full px-2.5 py-1.5 text-xs sm:text-sm border border-ln2 rounded-lg bg-well text-t1 focus:ring-2 focus:ring-ac focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
      />
    );
  }

  if (compact) {
    return (
      <div
        className="inline-flex items-center gap-1.5 text-[11px] text-t2"
      >
        <label
          htmlFor={inputId}
          className="shrink-0 cursor-pointer"
        >
          <span className="inline-flex items-center gap-1.5">
            <ParameterHelp label={displayLabel} description={description} tooltipId={descriptionId} />
            <span>{displayLabel}</span>
          </span>
        </label>
        <div className="min-w-16">{control}</div>
      </div>
    );
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        className="flex items-center justify-between text-sm font-medium text-t1 mb-1"
      >
        <span className="inline-flex items-center gap-1.5">
          <ParameterHelp label={displayLabel} description={description} tooltipId={descriptionId} />
          <span>{displayLabel}</span>
        </span>
      </label>
      {control}
      {description && <p className="text-[11px] text-t3 mt-1">{description}</p>}
    </div>
  );
}
