import React from 'react';
import { Ban, ChevronLeft, Pencil, Sparkles } from 'lucide-react';
import ServiceCard from './ServiceCard';
import CrossImageSuggestionCard from './CrossImageSuggestionCard';
import useAnnotationServices from './useAnnotationServices';
import useRailTools from './useRailTools';
import { PROMPT_ACTIONS, getPromptAction, getRailTool } from './toolModel';
import InstanceWarningModal from '../modals/InstanceWarningModal';
import { useToggleLeftDrawer } from '../../../stores/selectors/annotationSelectors';

const ACTION_ICONS = { Ban, Sparkles, Pencil };

/**
 * Contextual options drawer to the right of the tool rail.
 *
 * Hosts the three prompt actions — what happens once a prompt is placed — and
 * the annotation services that used to fill the left sidebar. The rail shows the
 * same three as icons; here they carry their names, because "Nothing" versus
 * "Add immediately" is a distinction an icon cannot make on its own.
 */
const ToolOptionsDrawer = () => {
  const toggleDrawer = useToggleLeftDrawer();
  const { railTool, promptAction, changePromptAction } = useRailTools();
  const {
    services,
    policyLoading,
    policyError,
    showInstanceWarning,
    closeInstanceWarning,
    confirmInstanceRun,
  } = useAnnotationServices();

  const toolName = getRailTool(railTool).name;

  return (
    <div className="w-[252px] flex-none flex flex-col bg-p1 border-r border-ln min-h-0">
      <div className="h-8 flex-none flex items-center gap-[7px] px-[10px] border-b border-ln">
        <span className="flex-1 text-sect font-bold tracking-[.09em] uppercase text-t3 truncate">
          {toolName} options
        </span>
        <button
          type="button"
          onClick={toggleDrawer}
          aria-label="Collapse tool options"
          className="w-[22px] h-[22px] flex items-center justify-center rounded-5 text-t3 hover:bg-hv hover:text-ac transition-colors duration-150"
        >
          <ChevronLeft size={14} strokeWidth={1.9} />
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-[10px] flex flex-col gap-[12px]">
        <div>
          <span className="block mb-[6px] text-sect font-bold tracking-[.09em] uppercase text-t3">
            When a prompt is placed
          </span>
          <div
            role="radiogroup"
            aria-label="What happens when a prompt is placed"
            className="flex items-center gap-[2px] p-[2px] rounded-9 border border-ln2 bg-well"
          >
            {PROMPT_ACTIONS.map((action) => {
              const Icon = ACTION_ICONS[action.icon];
              const active = promptAction === action.id;
              return (
                <button
                  key={action.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  aria-label={action.name}
                  onClick={() => changePromptAction(action.id)}
                  title={action.hint}
                  className={`flex-1 flex items-center justify-center gap-[5px] px-[6px] py-[6px] rounded-8 text-sect font-bold transition-colors duration-150 ${
                    active ? 'bg-acS text-ac' : 'text-t3 hover:bg-hv hover:text-t1'
                  }`}
                >
                  {Icon && <Icon size={13} strokeWidth={active ? 2.1 : 1.8} className="flex-none" />}
                  <span className="truncate">{action.tab}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-[6px] text-sect leading-[1.5] text-t3">
            {getPromptAction(promptAction).hint}
          </p>
          {railTool === 'point' && (
            <p className="mt-[6px] text-sect leading-[1.5] text-t3">
              Click to drop a point · drag to draw a box · right-click for a negative point.
            </p>
          )}
          {promptAction === 'manual' && (railTool === 'freehand' || railTool === 'polygon') && (
            <p className="mt-[6px] text-sect leading-[1.5] text-t3">
              {railTool === 'freehand'
                ? 'Press and drag to trace an outline · release to close and save it.'
                : 'Click to add points · double-click or Enter to close and save it.'}
            </p>
          )}
        </div>

        <div className="h-px bg-ln" />

        <div>
          <div className="flex items-center gap-[7px] mb-[9px]">
            <Sparkles size={14} className="text-ac flex-none" />
            <span className="text-row font-bold text-t1">Annotation services</span>
          </div>
          {policyLoading && (
            <p className="mb-[9px] px-[8px] py-[5px] rounded-6 bg-well2 text-ctl text-t2" role="status">
              Loading model routing policy…
            </p>
          )}
          {policyError && (
            <p className="mb-[9px] px-[8px] py-[5px] rounded-6 bg-errBg text-ctl text-err" role="alert">
              Unable to load model routing policy: {policyError}
            </p>
          )}
          <div className="flex flex-col gap-[9px]">
            {services
              .filter((s) => s.key === 'prompted' || s.key === 'instance')
              .map((service) => (
                <ServiceCard key={service.key} service={service} />
              ))}
            <CrossImageSuggestionCard />
          </div>
        </div>
      </div>

      <InstanceWarningModal
        isOpen={showInstanceWarning}
        onClose={closeInstanceWarning}
        onConfirm={confirmInstanceRun}
      />
    </div>
  );
};

export default ToolOptionsDrawer;
