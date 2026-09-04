import React, { useEffect, useState } from 'react';
import { Database, Brain, BarChart3, Tag, SquarePen, Download, Eye, GraduationCap, Users2, ClipboardCheck, Ruler, Wrench, Settings, HelpCircle, Wand2, Cpu } from 'lucide-react';
import ManagementCard from './ManagementCard';
import PhaseProgressBar from '../PhaseProgressBar';
import RoleBadge from '../RoleBadge';
import { usePermissions } from '../../../hooks/usePermissions';
import { Permission } from '../../../utils/permissions';
import { OVERALL_STATES, getPhase } from '../../../utils/imageStatus';
import { useGalleryStats } from '../../../stores/selectors';
import { fetchReviewSummary, fetchCorrectionSummary } from '../../../api/reviews';
import { DOCS, openDocs } from '../../../constants/docs';

/**
 * The sections the cards are grouped under, in display order. Each card names its
 * `group` by id; a group with no permitted cards for the current role is not
 * rendered at all (see `groupsInOrder` below).
 */
const GROUPS = [
  { id: 'annotation', title: 'Annotation workflow' },
  { id: 'data', title: 'Data & Analysis' },
  { id: 'models', title: 'Artificial Intelligence' },
  { id: 'configurations', title: 'Configurations' },
];

/**
 * The dataset's action grid.
 *
 * Each card declares the permission it needs; cards the current role cannot use
 * are left out entirely rather than shown greyed out — a reviewer has no way to
 * grant themselves upload rights, so a disabled "Data Management" tile would be
 * noise. Everything is still re-checked server-side.
 */
const ManagementCardsView = ({
  onDataManagementClick,
  onModelZooClick,
  onQuantificationsClick,
  onAnnotationClick,
  onLabelManagementClick,
  onExportCocoClick,
  onModelTrainingClick,
  onModelOrchestrationClick,
  onBatchInferenceClick,
  onBrowseAnnotations,
  onManageAccessClick,
  onCalibrateClick,
  onReviewClick,
  onCorrectClick,
  dataset,
}) => {
  const { can, canAny, role } = usePermissions(dataset);
  // Read from the store rather than taking a prop: this view is rendered as the
  // layout's children, so the stats the layout already fetched cannot be handed
  // down without threading them through DatasetGallery first.
  const stats = useGalleryStats();
  const canReview = can(Permission.REVIEW_APPROVE);
  const canCorrect = can(Permission.ANNOTATION_EDIT_OWN);

  // The pending-instance count on the Review card. Loaded here rather than in
  // the card so a failed fetch degrades to a card without a number, not an
  // empty tile.
  const [reviewSummary, setReviewSummary] = useState(null);
  useEffect(() => {
    if (!dataset?.id || !canReview) return undefined;
    let cancelled = false;
    fetchReviewSummary(dataset.id)
      .then((response) => {
        if (!cancelled && response?.success) setReviewSummary(response.summary);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [dataset?.id, canReview]);

  // The sent-back count on the Correct card. Loaded alongside the review summary,
  // and for the same reason kept here so a failed fetch degrades to a numberless
  // card rather than an empty tile.
  const [correctionSummary, setCorrectionSummary] = useState(null);
  useEffect(() => {
    if (!dataset?.id || !canCorrect) return undefined;
    let cancelled = false;
    fetchCorrectionSummary(dataset.id)
      .then((response) => {
        if (!cancelled && response?.success) setCorrectionSummary(response.summary);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [dataset?.id, canCorrect]);

  const canAnnotate = can(Permission.ANNOTATION_CREATE);
  // Calibration is a curator-level capability, and the two calibration kinds are
  // granted together, so either one is enough to make the card worth showing.
  const canCalibrate = canAny([Permission.PIXEL_SCALE_SET, Permission.CALIBRATION_SET]);

  // The Annotation card used to carry its own "N in progress, M not started"
  // subcaption from the annotation-queue summary. The phase bar below now says
  // exactly that, in the same units, so the fetch and the caption both went.

  // Per-phase progress, from the same store entry the dataset overview reads.
  // `total` is the image count and is shared by all three bars, so they compare.
  const phaseTotal =
    stats?.total ||
    OVERALL_STATES.reduce((acc, state) => acc + (stats?.overall?.[state.key] || 0), 0);

  /** The compact bar for a card's phase, or null for a card that has no phase. */
  const phaseBar = (phaseKey) => {
    const phase = phaseKey ? getPhase(phaseKey) : null;
    if (!phase || !phaseTotal) return null;
    return (
      <PhaseProgressBar
        phase={phase}
        counts={stats?.[phase.key]}
        total={phaseTotal}
        compact
      />
    );
  };

  const openRejections = correctionSummary?.open_rejections;
  const correctStat =
    openRejections == null
      ? null
      : openRejections === 1
      ? '1 instance sent back for correction'
      : openRejections > 1
      ? `${openRejections} instances sent back for correction`
      : 'Nothing sent back for correction';

  const pendingInstances = reviewSummary?.pending_instances;
  const reviewedInstances = reviewSummary?.reviewed_instances ?? 0;
  const reviewStat =
    pendingInstances == null
      ? null
      : pendingInstances === 1
      ? 'There is 1 instance to review'
      : pendingInstances > 1
      ? `There are ${pendingInstances} instances to review`
      : reviewedInstances > 0
      ? `${reviewedInstances} reviewed instance${
          reviewedInstances === 1 ? '' : 's'
        } available for re-review`
      : 'Nothing waiting for review right now';

  const cards = [
    {
      // First card of the workflow, matching the order of the phases themselves.
      // It opens the annotation canvas straight into its Calibrate tab rather than
      // being a page of its own: calibrating means looking at the image, at the
      // same zoom and pan as annotating it, which is why Calibrate is a mode there.
      id: 'calibrate',
      group: 'annotation',
      phase: 'calibrate',
      icon: Ruler,
      title: 'Calibrate',
      description: 'Set the scale, colour and intensity references so measurements compare across images',
      onClick: onCalibrateClick,
      permitted: canCalibrate,
      color: 'calibrate',
    },
    {
      id: 'annotation',
      group: 'annotation',
      phase: 'annotate',
      icon: SquarePen,
      title: 'Annotation',
      description: 'Annotate instances on images from this dataset',
      onClick: onAnnotationClick,
      permitted: canAnnotate,
      color: 'annotate',
    },
    {
      id: 'review',
      group: 'annotation',
      phase: 'review',
      icon: ClipboardCheck,
      title: 'Review',
      description: 'Go through annotated instances and approve them or send them back',
      stat: reviewStat,
      onClick: onReviewClick,
      permitted: canReview,
      color: 'review',
    },
    {
      id: 'correct',
      group: 'annotation',
      icon: Wrench,
      title: 'Correct',
      description: 'Correct sent back instances',
      stat: correctStat,
      onClick: onCorrectClick,
      permitted: canCorrect,
      color: 'rose',
    },
    {
      // The read-only counterpart for anyone who can see annotations but not make
      // them. Annotators and above reach the same view from inside the editor, so
      // this card only earns its space when the editor is unavailable.
      id: 'browse-annotations',
      group: 'annotation',
      icon: Eye,
      title: 'Browse Annotations',
      description: 'Look through the images and their annotations without editing anything',
      onClick: onBrowseAnnotations,
      permitted: can(Permission.ANNOTATION_READ) && !can(Permission.ANNOTATION_CREATE),
      color: 'teal',
    },
    {
      id: 'data-management',
      group: 'data',
      icon: Database,
      title: 'Data Management',
      description: 'Upload, organize, and manage your dataset images and files',
      onClick: onDataManagementClick,
      // Viewing the gallery only needs read; the upload/delete controls inside it
      // are gated separately.
      permitted: can(Permission.IMAGE_READ),
      color: 'blue',
    },
    {
      id: 'label-management',
      group: 'data',
      icon: Tag,
      title: 'Label Management',
      description: 'Create, edit, and organize labels and their hierarchical structure',
      onClick: onLabelManagementClick,
      permitted: can(Permission.LABEL_MANAGE),
      color: 'pink',
    },
    {
      id: 'export-coco',
      group: 'models',
      icon: Download,
      title: 'Export to COCO',
      description: 'Download the dataset in COCO format (with images or annotations only) for ML tasks',
      onClick: onExportCocoClick,
      permitted: can(Permission.EXPORT_ANNOTATIONS),
      color: 'amber',
    },
    {
      id: 'model-zoo',
      group: 'models',
      icon: Brain,
      title: 'Model Zoo',
      description: 'Browse, select, and manage AI models for training and inference',
      onClick: onModelZooClick,
      permitted: canAny([Permission.AI_INTERACTIVE, Permission.AI_BATCH_INFER]),
      color: 'purple',
    },
    {
      id: 'model-orchestration',
      group: 'models',
      icon: Cpu,
      title: 'Model Orchestration',
      description: 'Configure default models and per-label routing policies for interactive tools, cross-image suggestions, and batch runs',
      onClick: onModelOrchestrationClick,
      permitted: canAny([Permission.AI_INTERACTIVE, Permission.AI_BATCH_INFER]),
      color: 'indigo',
    },
    {
      id: 'model-training',
      group: 'models',
      icon: GraduationCap,
      title: 'Model Training',
      description: 'Train an instance segmentation model on this dataset and watch progress live',
      onClick: onModelTrainingClick,
      permitted: can(Permission.AI_TRAIN),
      color: 'indigo',
    },
    {
      id: 'batch-inference',
      group: 'models',
      icon: Wand2,
      title: 'Batch Inference',
      description: 'Let your models annotate the whole dataset — one model per label, run in hierarchy order',
      onClick: onBatchInferenceClick,
      permitted: can(Permission.AI_BATCH_INFER),
      color: 'purple',
    },
    {
      id: 'quantifications',
      group: 'data',
      icon: BarChart3,
      title: 'Quantifications',
      description: 'Analyze and quantify your annotated data with statistical insights',
      onClick: onQuantificationsClick,
      permitted: can(Permission.EXPORT_QUANTIFICATION),
      color: 'green',
    },
    {
      id: 'manage-access',
      group: 'configurations',
      icon: Users2,
      title: 'Manage Access',
      description: 'Control who can see, annotate and review this dataset, and how work is reviewed',
      onClick: onManageAccessClick,
      // Curators can see and invite; only owners can change roles. Either is
      // enough to make the page worth opening.
      permitted: canAny([Permission.MEMBER_LIST, Permission.MEMBER_GRANT, Permission.INVITE_CREATE]),
      color: 'teal',
    },
    {
      // Placeholder: will open a settings page. Dysfunctional for now.
      id: 'settings',
      group: 'configurations',
      icon: Settings,
      title: 'Settings',
      description: 'View and change all settings for this dataset',
      permitted: true,
      color: 'slate',
      disabled: true,
    },
    {
      // The documentation is a separate site, so this card leaves the app rather
      // than routing anywhere. It is permitted for everyone: whatever a role can
      // or cannot do here, reading about it should never be gated.
      id: 'help',
      group: 'configurations',
      icon: HelpCircle,
      title: 'Help',
      description: 'Read the documentation and learn how everything works',
      stat: 'Opens the documentation site in a new tab',
      onClick: () => openDocs(DOCS.home),
      permitted: true,
      color: 'blue',
    },
  ].filter((card) => card.permitted);

  // Cards are grouped into labelled sections so the grid reads as a few small
  // workflows instead of one long wall of tiles. A section with no permitted
  // cards for the current role is dropped entirely (header and divider included).
  const groupsInOrder = GROUPS.map((group) => ({
    ...group,
    cards: cards.filter((card) => card.group === group.id),
  })).filter((group) => group.cards.length > 0);

  return (
    <div className="overflow-y-auto bg-app p-4 sm:p-5 lg:p-6 h-full">
      <div className="w-full mx-auto">
        {/* The dataset's name and description used to live in the sidebar. This
            is the one page with room for them, and the one where knowing which
            dataset you are looking at actually decides what you click next. */}
        <div className="mb-6 sm:mb-8">
          <div className="flex items-center gap-3 mb-1 sm:mb-2">
            <h2 className="text-2xl sm:text-3xl font-bold text-t1">
              {dataset?.name || 'Dataset Management'}
            </h2>
            {role && <RoleBadge role={role} showDescription />}
          </div>
          <p className="text-sm sm:text-base text-t2 max-w-3xl">
            {dataset?.description || 'Manage your dataset, models, and annotations'}
          </p>
        </div>

        <div className="space-y-8 sm:space-y-10">
          {groupsInOrder.map((group) => (
            <section key={group.id}>
              <div className="flex items-center gap-4 mb-4 sm:mb-5">
                <h3 className="text-base sm:text-lg font-semibold text-t2 whitespace-nowrap">
                  {group.title}
                </h3>
                <div className="flex-1 h-px bg-hv2" />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 lg:gap-6">
                {group.cards.map((card) => (
                  <ManagementCard
                    key={card.id}
                    icon={card.icon}
                    title={card.title}
                    description={card.description}
                    stat={card.stat}
                    onClick={card.onClick}
                    color={card.color}
                    disabled={card.disabled}
                    progress={phaseBar(card.phase)}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ManagementCardsView;

