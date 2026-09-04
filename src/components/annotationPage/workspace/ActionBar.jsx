import React, { useEffect, useMemo, useState } from 'react';
import {
  Ban,
  Check,
  ChevronLeft,
  ChevronRight,
  Eraser,
  Layers,
  Loader2,
  Pencil,
  PenLine,
  RotateCcw,
  Shapes,
  Sparkles,
  SkipForward,
  Tag,
  Trash2,
  Wand2,
  X,
} from 'lucide-react';
import BarButton from './BarButton';
import LabelPicker from './LabelPicker';
import RejectMaskModal from '../modals/RejectMaskModal';
import useActionBarState from './useActionBarState';
import useObjectActions from './useObjectActions';
import useLabelAssignment from './useLabelAssignment';
import useAddShapesAsObjects from './useAddShapesAsObjects';
import useSupportedPromptTypes from './useSupportedPromptTypes';
import useSuggestSimilar from './useSuggestSimilar';
import useRailTools from './useRailTools';
import { getPromptAction } from './toolModel';
import { formatArea, getObjectDisplayName, getObjectState } from './objectViewModel';
import { resolveLabelColor } from './labelColorUtils';
import { getContourId } from '../../../utils/objectUtils';
import useAISegmentation from '../../../hooks/useAISegmentation';
import annotationSession from '../../../services/annotationSession';
import { useToast } from '../../../contexts/ToastContext';
import {
  useAIPrompts,
  useClearAllPrompts,
  useClearSelection,
  useSelectObject,
  useAvailablePromptedModels,
  usePromptedModel,
  useSetPromptedModel,
  usePicker,
  useSetPicker,
  useImageScale,
  useDatasetLabels,
  useLabelColorOverrides,
  useCurrentMaskId,
  useObjectsList,
  useUpdateObject,
  useRefinementModeActive,
} from '../../../stores/selectors/annotationSelectors';

/** Icons for the three prompt actions, keyed as PROMPT_ACTIONS names them. */
const ACTION_ICONS = { Ban, Sparkles, Pencil };

const RUN_COPY = {
  segment: 'Segmenting…',
  refine: 'Refining object…',
  suggest: 'Finding similar instances…',
  instance: 'Detecting instances…',
};

/** Small coloured dot that precedes the bar's context label. */
const Dot = ({ color, className = '' }) => (
  <span
    className={`w-[6px] h-[6px] rounded-full flex-none ${className}`}
    style={color ? { background: color } : undefined}
  />
);

/**
 * The dynamic action bar.
 *
 * One floating bar under the canvas that morphs to the current context. It
 * replaces the old in-canvas pills (`RunAIButton`, `SuggestSimilarButton`,
 * `AddAsObjectButton`) and surfaces the per-object actions that previously only
 * existed on row hover or in the context menu — both of which still work.
 *
 * See useActionBarState for how the state is derived, including why there is no
 * `draft` state on this backend.
 */
const ActionBar = () => {
  const bar = useActionBarState();
  const actions = useObjectActions();
  const labelling = useLabelAssignment();
  const suggest = useSuggestSimilar();
  const { isAdding, addShapes } = useAddShapesAsObjects();
  const supportedPromptTypes = useSupportedPromptTypes();
  const { promptAction, cyclePromptAction } = useRailTools();
  const { runSegmentation } = useAISegmentation();
  const { addToast } = useToast();

  const prompts = useAIPrompts();
  const clearAllPrompts = useClearAllPrompts();
  const clearSelection = useClearSelection();
  const selectObject = useSelectObject();
  const availableModels = useAvailablePromptedModels();
  const promptedModel = usePromptedModel();
  const setPromptedModel = useSetPromptedModel();
  const picker = usePicker();
  const setPicker = useSetPicker();
  const scale = useImageScale();
  const labels = useDatasetLabels();
  const colorOverrides = useLabelColorOverrides();
  const currentMaskId = useCurrentMaskId();
  const objects = useObjectsList();
  const updateObject = useUpdateObject();
  const refinementActive = useRefinementModeActive();

  const [query, setQuery] = useState('');
  const [reviewIndex, setReviewIndex] = useState(0);
  const [sendBackFor, setSendBackFor] = useState(null);

  const modelName =
    availableModels.find((model) => model.id === promptedModel)?.name || promptedModel || 'no model';

  /**
   * Whether the model can take what is on the canvas.
   *
   * This is where a model's declared prompt types belong now: the rail keeps
   * every shape available, because a polygon may be headed for "Add this
   * object" rather than for the model, and it is only this button that a
   * point/box-only model can actually refuse.
   */
  const unsupportedPrompt = useMemo(() => {
    if (!supportedPromptTypes) return null;
    const rejected = prompts.find((prompt) => supportedPromptTypes[prompt.type] === false);
    return rejected
      ? `${supportedPromptTypes.modelName} doesn’t accept ${rejected.type} prompts`
      : null;
  }, [prompts, supportedPromptTypes]);

  const runAIBlockedReason = !promptedModel ? 'Select a model first' : unsupportedPrompt;
  const canRunAI = !runAIBlockedReason;
  const addLabel = bar.addableCount > 1 ? `Add ${bar.addableCount} objects` : 'Add this object';

  const single = bar.selection.length === 1 ? bar.selection[0] : null;
  const reviewTarget = bar.reviewQueue[reviewIndex] || null;

  // Keep the queue index in range as objects are approved or removed.
  useEffect(() => {
    setReviewIndex((current) => Math.min(current, Math.max(bar.reviewQueue.length - 1, 0)));
  }, [bar.reviewQueue.length]);

  // The review queue drives the canvas selection so the current instance is framed.
  useEffect(() => {
    if (bar.state !== 'review' || !reviewTarget) return;
    selectObject(reviewTarget.id);
    actions.frameObject(reviewTarget);
    // `actions` is recreated each render; framing should follow the target only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bar.state, reviewTarget?.id]);

  const closePicker = () => {
    setPicker(null);
    setQuery('');
  };

  const labelItems = useMemo(() => {
    const target = picker === 'label' ? single || reviewTarget : bar.selection[0];
    return labelling.getLabelsForObject(target);
  }, [picker, single, reviewTarget, bar.selection, labelling]);

  const parentItems = useMemo(
    () =>
      objects
        .filter((object) => !bar.selection.some((selected) => selected.id === object.id))
        .map((object) => ({
          id: object.id,
          name: getObjectDisplayName(object),
          color: object.color,
        })),
    [objects, bar.selection]
  );

  const applyLabel = async (label) => {
    const targets =
      picker === 'label' && single
        ? [single]
        : bar.state === 'review' && reviewTarget
          ? [reviewTarget]
          : bar.selection;
    closePicker();
    await labelling.assignLabelToMany(targets, label);
  };

  /** Nests every selected object under the chosen parent. */
  const nestUnder = async (parent) => {
    closePicker();
    const parentContourId = getContourId(objects.find((object) => object.id === parent.id));
    if (parentContourId == null) {
      addToast({ type: 'error', message: 'That object cannot be a parent yet.' });
      return;
    }
    for (const object of bar.selection) {
      const contourId = getContourId(object);
      if (contourId == null) continue;
      try {
        // eslint-disable-next-line no-await-in-loop
        await annotationSession.modifyObject(contourId, { parent_id: parentContourId });
        updateObject(object.id, { parent_id: parent.id });
      } catch (error) {
        addToast({
          type: 'error',
          message: `Could not nest objects: ${error.message || 'not supported by the server'}`,
        });
        break;
      }
    }
  };

  const advanceReview = () =>
    setReviewIndex((current) => Math.min(current + 1, Math.max(bar.reviewQueue.length - 1, 0)));

  const acceptCurrent = async () => {
    if (!reviewTarget) return;
    if (getObjectState(reviewTarget) === 'unlabelled') {
      // Approval requires a class — open the picker rather than failing.
      setPicker('label');
      return;
    }
    try {
      await labelling.markReviewed(reviewTarget);
    } catch {
      // Already toasted; stay on the instance the server refused to approve.
      return;
    }
    advanceReview();
  };

  const rejectCurrent = async () => {
    if (!reviewTarget) return;
    try {
      await actions.remove(reviewTarget);
    } catch {
      // The action has already toasted; stay on the instance that failed.
      return;
    }
    advanceReview();
  };

  /**
   * The review verdict keys: `R` rejects the instance, `⏎` accepts it.
   *
   * They are bound here rather than in one of the shortcut hooks because the
   * review cursor is local state on this bar — a hook would have to duplicate it
   * to know what "the current instance" is. Neither key is contested while
   * reviewing: the rail no longer claims `R`, and useAnnotationKeyboardShortcuts
   * stands down from `⏎` in this mode.
   */
  useEffect(() => {
    if (bar.state !== 'review') return undefined;

    const handleKeyDown = (event) => {
      const isReject = event.key.toUpperCase() === 'R';
      const isAccept = event.key === 'Enter';
      if (!isReject && !isAccept) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      // Open overlays own the keyboard — the picker's own Enter would otherwise
      // both choose a label and approve.
      if (picker || sendBackFor) return;
      const target = event.target;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return;
      }
      if (!reviewTarget) return;
      event.preventDefault();
      if (isReject) rejectCurrent();
      else acceptCurrent();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // The handlers are recreated each render; the target they act on is the dep
    // that matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bar.state, picker, sendBackFor, reviewTarget?.id, bar.reviewQueue.length]);

  /**
   * `⇧⏎` commits the outlines on the canvas.
   *
   * Bound here because `addShapes` lives on this bar, and kept off plain `⏎` so
   * the two actions the bar offers keep distinct keys —
   * useAnnotationKeyboardShortcuts owns `⏎` for running the model.
   */
  useEffect(() => {
    if (bar.state !== 'prompt' || !bar.addableCount) return undefined;

    const handleKeyDown = (event) => {
      if (event.key !== 'Enter' || !event.shiftKey) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (picker) return;
      const target = event.target;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return;
      }
      event.preventDefault();
      addShapes();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [bar.state, bar.addableCount, picker, addShapes]);

  if (bar.state === 'editing') return null;

  // ---------------------------------------------------------------- contents

  let context = null;
  let sub = null;
  let buttons = null;
  let hint = null;

  if (bar.state === 'running') {
    context = (
      <>
        <Loader2 size={13} className="text-ac animate-spin flex-none" />
        <span className="text-btn font-bold text-t1 animate-dcScan">{RUN_COPY[bar.runKind]}</span>
      </>
    );
    sub = `${modelName}${prompts.length ? ` · ${prompts.length} prompts` : ''}`;
  } else if (bar.state === 'prompt') {
    context = (
      <>
        <Dot className="bg-ok" />
        <span className="text-btn font-bold text-t1">
          {bar.promptCount} {bar.promptCount === 1 ? 'prompt' : 'prompts'} placed
        </span>
      </>
    );
    sub = canRunAI ? modelName : runAIBlockedReason;
    buttons = (
      <>
        <BarButton icon={Eraser} label="Clear" shortcut="⌫" onClick={clearAllPrompts} />
        {bar.addableCount > 0 && (
          <BarButton
            icon={Shapes}
            label={isAdding ? 'Adding…' : addLabel}
            shortcut="⇧⏎"
            variant={canRunAI ? undefined : 'primary'}
            disabled={isAdding}
            title="Save the outline exactly as drawn, without running a model"
            onClick={addShapes}
          />
        )}
        <BarButton
          icon={Sparkles}
          label={refinementActive ? 'Refine object' : 'Run AI'}
          shortcut="⏎"
          variant={canRunAI ? 'primary' : undefined}
          disabled={!canRunAI}
          title={runAIBlockedReason || undefined}
          onClick={() => runSegmentation()}
        />
      </>
    );
    if (promptAction === 'ai') {
      hint = 'Every prompt runs the model as soon as it is placed.';
    }
  } else if (bar.state === 'object' && single) {
    const state = getObjectState(single);
    const area = formatArea(single, scale);
    const labelForObject = labels.find((label) => String(label.id) === String(single.labelId));
    context = (
      <>
        <Dot color={labelForObject ? resolveLabelColor(labelForObject, colorOverrides) : single.color} />
        <span className="text-btn font-bold text-t1 truncate max-w-[220px]">
          {getObjectDisplayName(single)}
        </span>
      </>
    );
    sub = area;
    buttons = (
      <>
        <BarButton icon={Wand2} label="Refine" onClick={() => actions.refine(single)} />
        <BarButton
          icon={Pencil}
          label="Edit contour"
          shortcut="E"
          onClick={() => actions.editContour(single)}
        />
        <BarButton icon={PenLine} label="Reshape" onClick={() => actions.reshapeByLine(single)} />
        <BarButton
          icon={Sparkles}
          label="Suggest similar"
          shortcut="2"
          disabled={!suggest.eligible}
          title={suggest.reason || undefined}
          onClick={suggest.run}
        />
        <BarButton
          icon={Trash2}
          label="Delete"
          shortcut="⌫"
          onClick={() => actions.remove(single)}
        />
        {state === 'unlabelled' ? (
          <BarButton
            icon={Tag}
            label="Assign label"
            shortcut="L"
            variant="primary"
            onClick={() => setPicker('label')}
          />
        ) : (
          <BarButton
            icon={Tag}
            label={single.label}
            shortcut="L"
            variant="chip"
            onClick={() => setPicker('label')}
          />
        )}
      </>
    );
  } else if (bar.state === 'multi') {
    context = (
      <>
        <Dot className="bg-ac" />
        <span className="text-btn font-bold text-t1">
          {bar.selection.length} objects selected
        </span>
      </>
    );
    buttons = (
      <>
        <BarButton icon={Layers} label="Group under…" onClick={() => setPicker('parent')} />
        <BarButton
          icon={Sparkles}
          label="Suggest similar"
          shortcut="2"
          disabled={!suggest.eligible}
          title={suggest.reason || undefined}
          onClick={suggest.run}
        />
        <BarButton
          icon={Trash2}
          label="Delete"
          shortcut="⌫"
          onClick={() => actions.removeMany(bar.selection)}
        />
        <BarButton icon={X} label="Clear selection" shortcut="esc" onClick={clearSelection} />
        <BarButton
          icon={Tag}
          label="Label all"
          shortcut="L"
          variant="primary"
          onClick={() => setPicker('label')}
        />
      </>
    );
  } else if (bar.state === 'review') {
    const total = bar.reviewQueue.length;
    context = (
      <>
        <Dot className="bg-rev" />
        <span className="text-btn font-bold text-t1">
          {total === 0 ? 'Nothing to review' : `Instance ${reviewIndex + 1} of ${total}`}
        </span>
      </>
    );
    sub = reviewTarget ? getObjectDisplayName(reviewTarget) : null;
    buttons = total === 0 ? null : (
      <>
        <BarButton
          icon={RotateCcw}
          label="Send back"
          disabled={!currentMaskId || !reviewTarget}
          onClick={() => setSendBackFor(reviewTarget)}
        />
        <BarButton
          icon={Tag}
          label="Relabel"
          shortcut="L"
          disabled={!reviewTarget}
          onClick={() => setPicker('label')}
        />
        <BarButton icon={SkipForward} label="Skip" onClick={advanceReview} />
        <BarButton
          icon={ChevronLeft}
          disabled={reviewIndex === 0}
          title="Previous instance"
          onClick={() => setReviewIndex((current) => Math.max(current - 1, 0))}
        />
        <BarButton
          icon={ChevronRight}
          disabled={reviewIndex >= total - 1}
          title="Next instance"
          onClick={advanceReview}
        />
        <BarButton
          icon={Trash2}
          label="Reject"
          shortcut="R"
          variant="danger"
          disabled={!reviewTarget}
          onClick={rejectCurrent}
        />
        <BarButton
          icon={Check}
          label="Accept"
          shortcut="⏎"
          variant="ok"
          disabled={!reviewTarget}
          onClick={acceptCurrent}
        />
      </>
    );
  } else {
    const action = getPromptAction(promptAction);
    const ActionIcon = ACTION_ICONS[action.icon] || Sparkles;
    context = (
      <>
        <Dot className={promptAction === 'nothing' ? 'bg-t3' : 'bg-ac'} />
        <span className="text-btn font-bold text-t1">Ready to annotate</span>
      </>
    );
    sub = action.short;
    buttons = (
      <>
        <BarButton
          icon={Sparkles}
          label={modelName}
          variant="chip"
          disabled={availableModels.length === 0}
          onClick={() => setPicker('model')}
        />
        <BarButton
          icon={ActionIcon}
          label={action.name}
          shortcut="A"
          variant="chip"
          title={action.hint}
          onClick={cyclePromptAction}
        />
      </>
    );
  }

  // z-70 keeps the bar above every canvas overlay: the focus dim (z40), the
  // prompt canvas as focus/refinement lift it (z45/z62) and the contour control
  // points (z65). Those overlays are full-bleed and pointer-events-auto, so
  // anything below them loses both the hover cursor and its clicks.
  return (
    <div className="absolute left-1/2 bottom-4 -translate-x-1/2 z-[70] w-max max-w-[calc(100%-28px)] flex flex-col items-center gap-[8px] pointer-events-none">
      {picker && (
        <div className="pointer-events-auto">
          {picker === 'model' ? (
            <LabelPicker
              items={availableModels.map((model) => ({ id: model.id, name: model.name }))}
              query={query}
              onQueryChange={setQuery}
              onSelect={(model) => {
                setPromptedModel(model.id);
                closePicker();
              }}
              onClose={closePicker}
              placeholder="Search models…"
              showColors={false}
              emptyMessage="No models available"
            />
          ) : picker === 'parent' ? (
            <LabelPicker
              items={parentItems}
              query={query}
              onQueryChange={setQuery}
              onSelect={nestUnder}
              onClose={closePicker}
              placeholder="Search objects…"
              caption="Nest the selected objects under…"
              emptyMessage="No other objects"
            />
          ) : (
            <LabelPicker
              items={labelItems}
              query={query}
              onQueryChange={setQuery}
              onSelect={applyLabel}
              onClose={closePicker}
              colorOverrides={colorOverrides}
              caption={
                labelling.getParentLabelName(single || reviewTarget || bar.selection[0])
                  ? `Sub-labels of ${labelling.getParentLabelName(single || reviewTarget || bar.selection[0])}`
                  : 'Root level'
              }
              emptyMessage="No labels valid at this level"
            />
          )}
        </div>
      )}

      <div className="pointer-events-auto flex flex-wrap items-center justify-center gap-[7px] p-[6px] rounded-12 bg-glass border border-ln2 shadow-bar backdrop-blur-sm">
        <div className="flex items-center gap-[6px] pl-[5px]">{context}</div>
        {sub && <span className="font-mono text-meta text-t3">{sub}</span>}
        {buttons && <div className="w-px h-5 bg-ln2" />}
        {buttons}
      </div>

      {hint && (
        <div className="pointer-events-none px-[9px] py-[3px] rounded-6 bg-glass text-sect text-t3">
          {hint}
        </div>
      )}

      {currentMaskId && sendBackFor && (
        <RejectMaskModal
          isOpen
          maskId={currentMaskId}
          contourId={getContourId(sendBackFor)}
          contourLabel={sendBackFor.label_name || sendBackFor.label}
          onClose={() => setSendBackFor(null)}
          onRejected={advanceReview}
        />
      )}
    </div>
  );
};

export default ActionBar;
