import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronUp, Loader2, Sparkles } from 'lucide-react';
import { useInferenceConfigSuggestion } from './useInferenceConfigSuggestion';
import {
  useActiveLabelId,
  useDatasetLabels,
  useSetActiveLabelId,
  useLabelColorOverrides,
} from '../../../stores/selectors/annotationSelectors';
import { resolveLabelColor } from './labelColorUtils';

/**
 * Annotation Service card for single-image cross-image AI suggestions.
 * Located in the tool-options drawer under Instance Segmentation.
 */
const CrossImageSuggestionCard = () => {
  const [open, setOpen] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);

  const labels = useDatasetLabels();
  const activeLabelId = useActiveLabelId();
  const setActiveLabelId = useSetActiveLabelId();
  const colorOverrides = useLabelColorOverrides();

  const {
    isLoadingConfig,
    configError,
    isConfigured,
    getResolvedBinding,
    isRunning,
    isAnyRunning,
    suggestLabel,
  } = useInferenceConfigSuggestion();

  // Selected label: currently armed/active label if it belongs to this dataset, otherwise fallback to first dataset label
  const isArmedValid = Array.isArray(labels) && labels.some((l) => Number(l.id) === Number(activeLabelId));
  const selectedLabelId = isArmedValid ? activeLabelId : (labels?.[0]?.id ?? null);
  const selectedLabel = labels.find((l) => Number(l.id) === Number(selectedLabelId)) || null;

  const resolved = useMemo(
    () => (selectedLabelId != null ? getResolvedBinding(selectedLabelId) : null),
    [getResolvedBinding, selectedLabelId]
  );

  const configured = Boolean(selectedLabelId && isConfigured(selectedLabelId));
  const running = Boolean(selectedLabelId && isRunning(selectedLabelId));

  const modelName = resolved?.model?.name || resolved?.binding?.model_registry_key || null;
  const color = selectedLabel ? resolveLabelColor(selectedLabel, colorOverrides) : '#888';

  return (
    <div className="rounded-9 border border-ln2 bg-well px-[10px] py-[9px]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-[7px] text-left"
      >
        <span
          className="w-[6px] h-[6px] rounded-full flex-none"
          style={{
            background: configured ? '#22c55e' : '#6d757d',
            boxShadow: `0 0 0 3px ${configured ? '#22c55e29' : '#6d757d29'}`,
          }}
          title={configured ? 'Suggestion model ready' : 'No suggestion model configured'}
        />
        <span className="flex-1 text-row font-bold text-t1 truncate">Cross-Image Suggestion</span>
        {running && <Loader2 size={13} className="text-ac animate-spin" />}
        {open ? (
          <ChevronUp size={13} strokeWidth={1.9} className="text-t3" />
        ) : (
          <ChevronDown size={13} strokeWidth={1.9} className="text-t3" />
        )}
      </button>

      {open && (
        <div className="mt-[8px] flex flex-col gap-[7px]">
          {isLoadingConfig ? (
            <div className="h-[27px] rounded-6 bg-hv animate-pulse" />
          ) : configError ? (
            <p className="px-[8px] py-[5px] rounded-6 bg-errBg text-ctl text-err">
              {configError}
            </p>
          ) : (
            <>
              {/* Target Label Picker */}
              <div>
                <label className="block text-meta font-semibold text-t3 uppercase tracking-wider mb-1">
                  Target Label
                </label>
                {labels.length > 0 ? (
                  <div className="flex items-center gap-1.5">
                    <span
                      className="w-[10px] h-[10px] rounded-[3px] flex-none border border-black/20"
                      style={{ background: color }}
                    />
                    <select
                      value={selectedLabelId || ''}
                      onChange={(e) => setActiveLabelId(Number(e.target.value))}
                      aria-label="Target label for suggestion"
                      className="flex-1 h-[27px] px-[8px] rounded-6 border border-ln2 bg-well2 text-row text-t1 outline-none focus:border-ac"
                    >
                      {labels.map((lbl) => (
                        <option key={lbl.id} value={lbl.id}>
                          {lbl.name}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <p className="text-sect text-t3">No labels created in dataset.</p>
                )}
              </div>

              {/* Model Info */}
              {selectedLabel && (
                <div className="flex items-center justify-between text-sect text-t2 px-1">
                  <span className="text-t3">Model:</span>
                  <span className="font-medium truncate max-w-[140px]" title={modelName || "Not configured"}>
                    {modelName || "None (configure in Orchestration)"}
                  </span>
                </div>
              )}

              {/* Action Button & Info */}
              <div className="flex items-center justify-between mt-[4px] gap-[8px]">
                <button
                  type="button"
                  onClick={() => suggestLabel(selectedLabelId)}
                  disabled={!configured || isAnyRunning || !selectedLabel}
                  aria-label={`Suggest ${selectedLabel?.name || 'label'}`}
                  className="inline-flex items-center gap-[6px] h-[26px] px-[10px] rounded-6 bg-accent text-onAccent text-row font-bold transition-[filter] hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {running ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : (
                    <Sparkles size={13} />
                  )}
                  <span>{running ? 'Suggesting…' : `Suggest ${selectedLabel?.name || ''}`}</span>
                </button>

                <button
                  type="button"
                  onClick={() => setInfoOpen((v) => !v)}
                  className="inline-flex items-center gap-[4px] h-[22px] px-[7px] rounded-5 text-meta font-semibold text-t3 hover:bg-hv hover:text-t2 transition-colors"
                >
                  Info
                  {infoOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </button>
              </div>

              {infoOpen && (
                <div className="mt-[4px] pt-[6px] border-t border-ln text-sect leading-[1.5] text-t3">
                  Runs the model configured in Model Orchestration for {selectedLabel?.name || 'the target label'}, retrieving annotated exemplars across the dataset and patching candidate predictions onto the active image.
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default CrossImageSuggestionCard;
