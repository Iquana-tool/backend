import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  BookOpen,
  Bug,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Keyboard,
  LogOut,
  Maximize,
  Menu as MenuIcon,
  Minus,
  Moon,
  PanelLeft,
  PanelRight,
  Plus,
  Redo2,
  Settings,
  Sun,
  Tag,
  Trash2,
  Undo2,
  User,
  Download,
} from 'lucide-react';
import Menu, { MenuDivider, MenuHeader, MenuItem } from './primitives/Menu';
import ToolbarButton from './primitives/ToolbarButton';
import Tooltip from './primitives/Tooltip';
import ConfirmDialog from './ConfirmDialog';
import useWorkspaceImageNav from './useWorkspaceImageNav';
import useImageLevelActions from './useImageLevelActions';
import { useDataset } from '../../../contexts/DatasetContext';
import { useAuth } from '../../../contexts/AuthContext';
import { PHASES, getStateDescriptor } from '../../../utils/imageStatus';
import { MAX_ZOOM, MIN_ZOOM, ZOOM_STEP, clampZoom } from './constants';
import { CHIP_MODE_LABELS } from '../canvas/chipLayout';
import {
  useAnnotationStatus,
  usePhaseStatus,
  useZoomLevel,
  useSetZoomLevel,
  useChipMode,
  useCycleChipMode,
  useSetPanOffset,
  useWorkspaceMode,
  useSetWorkspaceMode,
  useWorkspaceTheme,
  useToggleTheme,
  useLeftDrawerOpen,
  useToggleLeftDrawer,
  useRightPanelOpen,
  useToggleRightPanel,
  useUndoActionLabel,
  useRedoActionLabel,
  useSetShortcutSheetOpen,
  useSetRightTab,
} from '../../../stores/selectors/annotationSelectors';
import useAnnotationHistory from '../../../hooks/useAnnotationHistory';
import { DOCS, openDocs } from '../../../constants/docs';

/** Status-pill tokens for the three workflow states. */
const STATUS_TONE = {
  not_started: 'bg-well text-t3',
  in_progress: 'bg-warnBg text-warn',
  finished: 'bg-okBg text-ok',
};

const Divider = () => <div className="w-px h-[18px] bg-ln flex-none" />;

/** Inset well that groups related toolbar controls (undo/redo, zoom, mode). */
const Group = ({ children }) => (
  <div className="flex items-center gap-[2px] p-[2px] rounded-7 bg-well flex-none">{children}</div>
);

const TopToolbar = () => {
  const navigate = useNavigate();
  const { currentDataset } = useDataset();
  const { user, logout } = useAuth();

  const [appMenuOpen, setAppMenuOpen] = useState(false);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [confirmRemoveAll, setConfirmRemoveAll] = useState(false);

  const nav = useWorkspaceImageNav();
  const annotationStatus = useAnnotationStatus();
  const phaseStatus = usePhaseStatus();
  const overall = getStateDescriptor(annotationStatus);

  const zoomLevel = useZoomLevel();
  const setZoomLevel = useSetZoomLevel();
  const chipMode = useChipMode();
  const cycleChipMode = useCycleChipMode();
  const setPanOffset = useSetPanOffset();

  const mode = useWorkspaceMode();
  const setMode = useSetWorkspaceMode();
  const theme = useWorkspaceTheme();
  const toggleTheme = useToggleTheme();
  const leftDrawerOpen = useLeftDrawerOpen();
  const toggleLeftDrawer = useToggleLeftDrawer();
  const rightPanelOpen = useRightPanelOpen();
  const toggleRightPanel = useToggleRightPanel();
  const setRightTab = useSetRightTab();
  const setShortcutSheetOpen = useSetShortcutSheetOpen();

  // Undo/redo covers two stacks. Unsubmitted prompt dots are undone on the client
  // (they never reached the server); everything the annotator has actually done to
  // the objects is undone through the server's action history, which is the only
  // place a deleted object can be brought back with its own id and children.
  // useAnnotationHistory decides which of the two a press means — see the hook.
  const history = useAnnotationHistory();

  // Naming the step makes the button honest about what it will revert — "Undo
  // delete object" reads very differently from a generic "Undo" when the thing
  // you are about to lose is a suggestion run.
  const nextUndo = useUndoActionLabel();
  const nextRedo = useRedoActionLabel();
  const undoLabel = history.undoSource === 'prompt'
    ? 'Undo prompt'
    : (history.undoSource ? `Undo ${nextUndo || 'last action'}` : 'Nothing to undo');
  const redoLabel = history.redoSource === 'prompt'
    ? 'Redo prompt'
    : (history.redoSource ? `Redo ${nextRedo || 'last action'}` : 'Nothing to redo');

  const imageActions = useImageLevelActions();

  const zoomTo = (level) => setZoomLevel(clampZoom(level));
  const resetView = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const backToGallery = () => {
    navigate(
      currentDataset ? `/dataset/${currentDataset.id}/datamanagement` : '/datasets'
    );
  };

  const initial = (user?.username || '?').charAt(0).toUpperCase();

  return (
    <div className="h-[44px] flex-none flex items-center gap-[10px] px-[10px] bg-p1 border-b border-ln">
      {/* App menu + wordmark */}
      <div className="relative flex items-center gap-[8px] flex-none">
        <button
          type="button"
          onClick={() => {
            setAccountMenuOpen(false);
            setAppMenuOpen((open) => !open);
          }}
          aria-label="Main menu"
          aria-expanded={appMenuOpen}
          className="w-7 h-7 flex items-center justify-center rounded-6 bg-hv text-t1 hover:bg-hv2 transition-colors"
        >
          <MenuIcon size={15} strokeWidth={1.7} />
        </button>
        <span className="text-[11px] font-bold tracking-[.12em] text-t2 select-none">IQUANA</span>

        <Menu open={appMenuOpen} onClose={() => setAppMenuOpen(false)}>
          <MenuItem
            icon={ArrowLeft}
            label="Back to gallery"
            shortcut="⌘←"
            onClick={() => {
              setAppMenuOpen(false);
              backToGallery();
            }}
          />
          {/* The design calls for an export entry here, but the only export
              helper in the client (api/downloads.js) is unreachable dead code
              with a malformed URL and no file-save step. Surfaced disabled
              rather than silently omitted or wired to something broken. */}
          <MenuItem
            icon={Download}
            label="Export annotations…"
            disabled
            title="Export is not available from the workspace yet — use the dataset manager"
          />
          <MenuItem
            icon={Keyboard}
            label="Keyboard shortcuts"
            shortcut="?"
            onClick={() => {
              setAppMenuOpen(false);
              setShortcutSheetOpen(true);
            }}
          />
          <MenuItem
            icon={BookOpen}
            label="Documentation"
            onClick={() => {
              setAppMenuOpen(false);
              openDocs(DOCS.drawAndRefine);
            }}
          />
          <MenuItem
            icon={Bug}
            label="Report a bug"
            onClick={() => {
              setAppMenuOpen(false);
              window.open(
                'https://github.com/Iquana-tool/iquana-tool/issues',
                '_blank',
                'noopener,noreferrer'
              );
            }}
          />
          <MenuDivider />
          <MenuItem
            icon={Trash2}
            label="Remove all annotations"
            danger
            disabled={!imageActions.hasMask}
            onClick={() => {
              setAppMenuOpen(false);
              setConfirmRemoveAll(true);
            }}
          />
        </Menu>
      </div>

      <Divider />

      {/* Breadcrumb */}
      <div className="flex items-center gap-[7px] min-w-0 whitespace-nowrap">
        <button
          type="button"
          onClick={backToGallery}
          className="flex items-center gap-[5px] h-[26px] px-[8px] rounded-6 text-btn text-t2 hover:bg-hv hover:text-t1 transition-colors flex-none"
        >
          <ArrowLeft size={13} strokeWidth={1.9} />
          <span>Gallery</span>
        </button>
        <span className="text-t3 flex-none">/</span>
        <span className="text-btn text-t2 truncate max-w-[180px]">
          {currentDataset?.name || 'Loading…'}
        </span>
        <span className="text-t3 flex-none">/</span>
        <span className="text-btn font-semibold text-t1 truncate max-w-[180px]">
          {nav.currentImage?.name || '—'}
        </span>
        {/* Combined status, with the three phases behind it on hover. Three pills
            in the breadcrumb would crowd out the image name; one plus a tooltip
            still lets the annotator see which phase is holding the image back. */}
        <span
          title={PHASES.map(
            (phase) => `${phase.label}: ${getStateDescriptor(phaseStatus?.[phase.key]).label}`
          ).join('\n')}
          className={`inline-flex items-center gap-[4px] h-[19px] px-[7px] rounded-5 text-sect font-semibold flex-none ${
            STATUS_TONE[overall.key] || STATUS_TONE.not_started
          }`}
        >
          {overall.key === 'finished' && <Check size={13} strokeWidth={2} />}
          {overall.label}
        </span>

        {/* Per-phase dots: the compact form of the same tooltip, so the phase a
            reviewer sent back is visible without hovering. Each dot is a tone of
            its own phase's hue, matching the mode tabs and the strip on the
            gallery thumbnails — position and colour say the same thing. */}
        <span className="inline-flex items-center gap-[3px] flex-none">
          {PHASES.map((phase) => {
            const state = getStateDescriptor(phaseStatus?.[phase.key]);
            return (
              <span
                key={phase.key}
                title={`${phase.label}: ${state.label}`}
                className={`w-[6px] h-[6px] rounded-full ${phase.fill[state.key]}`}
              />
            );
          })}
        </span>
      </div>

      {/* Image navigation */}
      <div className="flex items-center gap-[3px] ml-[6px] flex-none">
        <ToolbarButton
          icon={ChevronLeft}
          label="Previous image"
          shortcut="←"
          disabled={!nav.canGoPrev}
          onClick={nav.goPrev}
        />
        <span className="font-mono text-ctl text-t3 tabular-nums">
          {nav.total > 0 ? `${nav.currentIndex + 1} / ${nav.total}` : '– / –'}
        </span>
        <ToolbarButton
          icon={ChevronRight}
          label="Next image"
          shortcut="→"
          disabled={!nav.canGoNext}
          onClick={nav.goNext}
        />
      </div>

      <div className="flex-1 min-w-[8px]" />

      <Group>
        <ToolbarButton
          icon={Undo2}
          label={undoLabel}
          shortcut="⌘Z"
          disabled={!history.canUndo || history.busy}
          onClick={history.undo}
        />
        <ToolbarButton
          icon={Redo2}
          label={redoLabel}
          shortcut="⇧⌘Z"
          disabled={!history.canRedo || history.busy}
          onClick={history.redo}
        />
      </Group>

      <Group>
        <ToolbarButton
          icon={Minus}
          label="Zoom out"
          shortcut="−"
          disabled={zoomLevel <= MIN_ZOOM}
          onClick={() => zoomTo(zoomLevel / ZOOM_STEP)}
        />
        <Tooltip label="Fit to screen" shortcut="0" placement="bottom">
          <button
            type="button"
            onClick={resetView}
            className="min-w-[48px] h-[26px] rounded-5 font-mono text-ctl text-t1 hover:bg-hv2 transition-colors tabular-nums"
          >
            {Math.round(zoomLevel * 100)}%
          </button>
        </Tooltip>
        <ToolbarButton
          icon={Plus}
          label="Zoom in"
          shortcut="+"
          disabled={zoomLevel >= MAX_ZOOM}
          onClick={() => zoomTo(zoomLevel * ZOOM_STEP)}
        />
        <ToolbarButton icon={Maximize} label="Reset view" onClick={resetView} />
        {/* Sits with zoom because it is the same job: making a crowded image
            readable. Cycles all → selected only → off. */}
        <ToolbarButton
          icon={Tag}
          label={CHIP_MODE_LABELS[chipMode]}
          shortcut="T"
          active={chipMode !== 'all'}
          onClick={cycleChipMode}
        />
      </Group>

      {/* Calibrate / Annotate / Review.
          Three views of the same image sharing one canvas and one viewport —
          switching keeps the zoom, the pan and the selection, which is the whole
          reason calibration is a mode here rather than a separate page.

          Each mode wears its phase's hue, and the ring around the canvas below
          follows it, so the mode is legible from the stage without looking back up
          at this control. Inactive tabs keep the hue at reduced opacity, which
          makes the group read as a key for the colour that surrounds the image. */}
      <Group>
        {PHASES.map((phase) => {
          const active = mode === phase.key;
          return (
            <button
              key={phase.key}
              type="button"
              onClick={() => setMode(phase.key)}
              aria-pressed={active}
              className={`h-[26px] px-[11px] rounded-5 text-btn font-semibold transition-colors ${
                active
                  ? `${phase.bg2} ${phase.text}`
                  : `${phase.text} opacity-55 hover:opacity-100 hover:bg-hv2`
              }`}
            >
              {phase.label}
            </button>
          );
        })}
      </Group>

      <ToolbarButton
        icon={theme === 'dark' ? Sun : Moon}
        label={theme === 'dark' ? 'Light theme' : 'Dark theme'}
        onClick={toggleTheme}
      />
      <ToolbarButton
        icon={PanelLeft}
        label="Tool options"
        shortcut="⌥1"
        active={leftDrawerOpen}
        onClick={toggleLeftDrawer}
      />
      <ToolbarButton
        icon={PanelRight}
        label="Side panel"
        shortcut="⌥2"
        active={rightPanelOpen}
        onClick={toggleRightPanel}
      />
      <ToolbarButton
        icon={Settings}
        label="Annotation services"
        onClick={() => {
          if (!leftDrawerOpen) toggleLeftDrawer();
        }}
      />

      {/* Account */}
      <div className="relative flex-none">
        <button
          type="button"
          onClick={() => {
            setAppMenuOpen(false);
            setAccountMenuOpen((open) => !open);
          }}
          aria-expanded={accountMenuOpen}
          className="flex items-center gap-[5px] h-[26px] pl-[3px] pr-[7px] rounded-14 bg-hv hover:bg-hv2 transition-colors"
        >
          <span className="w-5 h-5 rounded-full bg-acS text-ac text-sect font-bold flex items-center justify-center">
            {initial}
          </span>
          <ChevronDown size={13} strokeWidth={1.9} className="text-t2" />
        </button>

        <Menu
          open={accountMenuOpen}
          onClose={() => setAccountMenuOpen(false)}
          align="right"
          width={200}
        >
          <MenuHeader
            title={user?.username || 'Signed out'}
            subtitle={currentDataset?.name}
          />
          {/* No account-settings route exists in this app yet. */}
          <MenuItem
            icon={User}
            label="Account settings"
            disabled
            title="Account settings are not available yet"
          />
          {imageActions.isReviewable ? (
            <MenuItem
              icon={Check}
              label="Unmark as fully annotated"
              disabled={imageActions.isProcessing || !imageActions.hasMask}
              onClick={() => {
                setAccountMenuOpen(false);
                imageActions.unmarkAsFullyAnnotated();
              }}
            />
          ) : (
            <MenuItem
              icon={Check}
              label="Mark as fully annotated"
              disabled={imageActions.isProcessing || !imageActions.hasMask}
              onClick={() => {
                setAccountMenuOpen(false);
                imageActions.markAsFullyAnnotated();
              }}
            />
          )}
          <MenuDivider />
          <MenuItem
            icon={LogOut}
            label="Log out"
            onClick={() => {
              setAccountMenuOpen(false);
              logout();
              navigate('/');
            }}
          />
        </Menu>
      </div>

      <ConfirmDialog
        open={confirmRemoveAll}
        title="Remove all annotations?"
        body="Every object on this image is deleted, including any that have already been reviewed. This cannot be undone."
        confirmLabel="Remove all"
        busy={imageActions.isProcessing}
        onCancel={() => setConfirmRemoveAll(false)}
        onConfirm={async () => {
          await imageActions.removeAllAnnotations();
          setConfirmRemoveAll(false);
          // The count in the panel header is derived, so nothing else to reset.
          setRightTab('objects');
        }}
      />
    </div>
  );
};

export default TopToolbar;
