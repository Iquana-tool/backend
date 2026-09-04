import React, { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Eye, EyeOff, Plus } from 'lucide-react';
import { resolveLabelColor, SWATCHES } from './labelColorUtils';
import { createLabel, fetchLabels } from '../../../api/labels';
import { extractLabelsFromResponse } from '../../../utils/labelHierarchy';
import { useDataset } from '../../../contexts/DatasetContext';
import { useToast } from '../../../contexts/ToastContext';
import {
  useDatasetLabels,
  useSetDatasetLabels,
  useObjectsList,
  useObjectsVisibility,
  useToggleVisibility,
  useActiveLabelId,
  useSetActiveLabelId,
  useLabelColorOverrides,
  useSetLabelColorOverride,
} from '../../../stores/selectors/annotationSelectors';

/** Nests the flat label list, then flattens it again in display order. */
const buildRows = (labels, collapsed, depth = 0, parentId = null, out = []) => {
  const children = labels.filter((label) => {
    const labelParent = label.parent_id ?? null;
    return String(labelParent ?? '') === String(parentId ?? '');
  });

  for (const label of children) {
    const hasChildren = labels.some(
      (other) => String(other.parent_id ?? '') === String(label.id)
    );
    out.push({ label, depth, hasChildren });
    if (hasChildren && !collapsed[label.id]) {
      buildRows(labels, collapsed, depth + 1, label.id, out);
    }
  }
  return out;
};

/**
 * Labels tab — the class taxonomy.
 *
 * Clicking a row arms that class for the next annotation (shown by the ACTIVE
 * pill, the rail swatch and the status bar). The eye toggles the same
 * label-level visibility the Objects tab's Filter mode uses, so hiding a class
 * here hides it on the canvas too.
 *
 * Colours are a client-side override: the label API has no colour field, so a
 * change here is local to this browser and does not travel with the dataset.
 */
const LabelsTab = () => {
  const labels = useDatasetLabels();
  const setDatasetLabels = useSetDatasetLabels();
  const objects = useObjectsList();
  const visibility = useObjectsVisibility();
  const toggleVisibility = useToggleVisibility();
  const activeLabelId = useActiveLabelId();
  const setActiveLabelId = useSetActiveLabelId();
  const colorOverrides = useLabelColorOverrides();
  const setLabelColorOverride = useSetLabelColorOverride();

  const { currentDataset } = useDataset();
  const { addToast } = useToast();


  const [collapsed, setCollapsed] = useState({});
  const [swatchFor, setSwatchFor] = useState(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [busy, setBusy] = useState(false);

  const rows = useMemo(() => buildRows(labels, collapsed), [labels, collapsed]);

  const armedLabel = labels.find((label) => String(label.id) === String(activeLabelId));

  // Instance counts per class, so the tree shows what is actually on the image.
  const counts = useMemo(() => {
    const map = new Map();
    for (const object of objects) {
      if (object.labelId == null) continue;
      const key = String(object.labelId);
      map.set(key, (map.get(key) || 0) + 1);
    }
    return map;
  }, [objects]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name || !currentDataset) return;
    setBusy(true);
    try {
      await createLabel({ name, parent_id: null }, currentDataset.id);
      // Refetch rather than patch: the server assigns the id and value.
      const refreshed = extractLabelsFromResponse(await fetchLabels(currentDataset.id));
      const map = new Map();
      refreshed.forEach((label) => {
        if (label?.id && label?.name) {
          map.set(Number(label.id), label.name);
          map.set(String(label.id), label.name);
        }
      });
      setDatasetLabels(refreshed, map);
      setNewName('');
      setCreating(false);
      addToast({ type: 'success', message: `Label “${name}” created.` });
    } catch (error) {
      addToast({
        type: 'error',
        message: `Could not create label: ${error.message || 'Unknown error'}`,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="flex items-center gap-[6px] h-[26px] px-[8px] flex-none">
        <span className="text-sect font-bold tracking-[.08em] uppercase text-t3">
          Label taxonomy
        </span>
        <span className="flex-1" />
        <span className="text-meta text-t3">click to arm</span>
      </div>

      {armedLabel && (
        <div className="mx-[8px] mb-[7px] flex items-center gap-[6px] px-[8px] py-[6px] rounded-7 bg-acS border border-acLn flex-none">
          <span className="text-meta text-ac leading-[1.4]">
            New objects are labelled <strong className="font-bold">{armedLabel.name}</strong> automatically.
          </span>
          <button
            type="button"
            onClick={() => setActiveLabelId(armedLabel.id)}
            className="ml-auto flex-none text-meta font-bold text-ac underline underline-offset-2 hover:brightness-125"
          >
            Disarm
          </button>
        </div>
      )}



      <div className="flex-1 min-h-0 overflow-y-auto px-[8px] pb-[10px]">
        {rows.length === 0 ? (
          <p className="py-[20px] text-center text-meta text-t3">
            This dataset has no labels yet.
          </p>
        ) : (
          rows.map(({ label, depth, hasChildren }) => {
            const armed = String(activeLabelId) === String(label.id);
            const color = resolveLabelColor(label, colorOverrides);
            const visible = visibility.labels[String(label.id)] !== false;
            const count = counts.get(String(label.id)) || 0;

            return (
              <div key={label.id} className="relative">
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => setActiveLabelId(label.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setActiveLabelId(label.id);
                    }
                  }}
                  className={`flex items-center gap-[5px] h-[27px] pr-[4px] rounded-6 cursor-pointer transition-colors ${
                    armed ? 'bg-acS shadow-[inset_0_0_0_1px_var(--acLn)]' : 'hover:bg-hv'
                  }`}
                  style={{ paddingLeft: 2 + depth * 14 }}
                >
                  <button
                    type="button"
                    aria-label={collapsed[label.id] ? 'Expand' : 'Collapse'}
                    onClick={(e) => {
                      e.stopPropagation();
                      setCollapsed((current) => ({
                        ...current,
                        [label.id]: !current[label.id],
                      }));
                    }}
                    className={`flex-none text-t3 hover:text-t1 transition-colors ${
                      hasChildren ? '' : 'opacity-0 pointer-events-none'
                    }`}
                  >
                    {collapsed[label.id] ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                  </button>

                  <button
                    type="button"
                    title="Change colour"
                    aria-label={`Change colour of ${label.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSwatchFor(swatchFor === label.id ? null : label.id);
                    }}
                    className="w-[11px] h-[11px] rounded-[3px] flex-none border border-black/20"
                    style={{ background: color }}
                  />

                  <span
                    className={`flex-1 min-w-0 truncate text-row ${
                      armed ? 'font-semibold text-ac' : 'text-t2'
                    }`}
                  >
                    {label.name}
                  </span>

                  {armed && (
                    <span className="inline-flex items-center h-[15px] px-[5px] rounded-4 bg-acS text-ac text-badge font-bold tracking-[.04em] flex-none">
                      ACTIVE
                    </span>
                  )}

                  <span className="font-mono text-meta text-t3 tabular-nums flex-none">
                    {count}
                  </span>



                  <button
                    type="button"
                    aria-label={visible ? `Hide ${label.name}` : `Show ${label.name}`}
                    title={
                      visible
                        ? 'Hide objects with this label (Filter mode)'
                        : 'Show objects with this label'
                    }
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleVisibility(label.id);
                    }}
                    className="w-5 h-5 flex-none flex items-center justify-center rounded-5 text-t3 hover:bg-hv hover:text-ac transition-colors duration-150"
                  >
                    {visible ? <Eye size={13} /> : <EyeOff size={13} />}
                  </button>
                </div>

                {swatchFor === label.id && (
                  <>
                    <div className="fixed inset-0 z-[70]" onClick={() => setSwatchFor(null)} />
                    <div
                      className="absolute left-[20px] top-[26px] z-[80] grid grid-cols-6 gap-[4px] p-[7px] rounded-8 bg-p2 border border-ln2 shadow-picker animate-dcPop"
                      style={{ width: 'max-content' }}
                    >
                      {SWATCHES.map((swatch) => (
                        <button
                          key={swatch}
                          type="button"
                          aria-label={`Use ${swatch}`}
                          onClick={() => {
                            setLabelColorOverride(label.id, swatch);
                            setSwatchFor(null);
                          }}
                          className="w-4 h-4 rounded-[3px] border border-black/20 transition-transform hover:scale-110"
                          style={{ background: swatch }}
                        />
                      ))}
                    </div>
                  </>
                )}
              </div>
            );
          })
        )}
      </div>

      <div className="flex-none p-[8px] border-t border-ln">
        {creating ? (
          <div className="flex items-center gap-[5px]">
            <input
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreate();
                if (e.key === 'Escape') setCreating(false);
              }}
              placeholder="Label name"
              aria-label="New label name"
              className="flex-1 min-w-0 h-7 px-[8px] rounded-7 bg-well border border-ln2 text-row text-t1 outline-none focus:border-ac placeholder:text-t3"
            />
            <button
              type="button"
              onClick={handleCreate}
              disabled={busy || !newName.trim()}
              className="h-7 px-[10px] rounded-7 bg-accent text-onAccent text-btn font-bold disabled:opacity-40"
            >
              Add
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setCreating(true)}
            disabled={!currentDataset}
            className="w-full h-7 flex items-center justify-center gap-[6px] rounded-7 border border-dashed border-ln2 text-btn text-t2 hover:text-t1 hover:border-ac transition-colors disabled:opacity-40"
          >
            <Plus size={13} />
            New label
          </button>
        )}
      </div>
    </div>
  );
};

export default LabelsTab;
