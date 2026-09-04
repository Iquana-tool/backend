import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  BarChart3,
  Brain,
  ChevronDown,
  ClipboardCheck,
  Cpu,
  Database,
  Eye,
  GraduationCap,
  Ruler,
  SquarePen,
  Tag,
  Users2,
  Wand2,
  Wrench,
} from "lucide-react";
import { usePermissions } from "../../../hooks/usePermissions";
import { Permission } from "../../../utils/permissions";

/**
 * Where a dataset can be navigated to, grouped the way the overview cards are.
 *
 * This is the navigation the left sidebar used to stand in for: every dataset
 * page is now one click away from every other one, so a view no longer has to be
 * entered from the overview grid and left again through a "Back" button.
 *
 * The groups mirror `ManagementCardsView` deliberately — same headings, same
 * icons — so the menu reads as the card grid folded up. Only destinations live
 * here: the cards also carry actions that open a modal (Export to COCO, the
 * annotation queue builder), which have no URL to point at.
 */
const NAV_GROUPS = [
  {
    id: "annotation",
    label: "Annotation",
    items: [
      {
        id: "calibrate",
        label: "Calibrate",
        description: "Set scale, colour and intensity references",
        icon: Ruler,
        path: (datasetId) => `/dataset/${datasetId}/annotate?mode=calibrate`,
        permitted: ({ canAny }) =>
          canAny([Permission.PIXEL_SCALE_SET, Permission.CALIBRATION_SET]),
      },
      {
        id: "annotate",
        label: "Annotate",
        description: "Open the editor on this dataset",
        icon: SquarePen,
        path: (datasetId) => `/dataset/${datasetId}/annotate`,
        permitted: ({ can }) => can(Permission.ANNOTATION_CREATE),
      },
      {
        id: "review",
        label: "Review",
        description: "Approve annotated instances or send them back",
        icon: ClipboardCheck,
        path: (datasetId) => `/dataset/${datasetId}/review`,
        permitted: ({ can }) => can(Permission.REVIEW_APPROVE),
      },
      {
        id: "correct",
        label: "Correct",
        description: "Work through instances sent back to you",
        icon: Wrench,
        path: (datasetId) => `/dataset/${datasetId}/correct`,
        permitted: ({ can }) => can(Permission.ANNOTATION_EDIT_OWN),
      },
      {
        // The read-only counterpart, shown only when the editor is unavailable —
        // everyone else reaches the same view from inside the editor.
        id: "browse",
        label: "Browse Annotations",
        description: "Look through images and annotations read-only",
        icon: Eye,
        path: (datasetId) => `/dataset/${datasetId}/view`,
        permitted: ({ can }) =>
          can(Permission.ANNOTATION_READ) && !can(Permission.ANNOTATION_CREATE),
      },
    ],
  },
  {
    id: "data",
    label: "Data & Analysis",
    items: [
      {
        id: "images",
        label: "Data Management",
        description: "Upload, filter and organise the images",
        icon: Database,
        path: (datasetId) => `/dataset/${datasetId}/datamanagement/images`,
        permitted: ({ can }) => can(Permission.IMAGE_READ),
      },
      {
        id: "labels",
        label: "Label Management",
        description: "Edit the label space and its hierarchy",
        icon: Tag,
        path: (datasetId) => `/dataset/${datasetId}/datamanagement/labels`,
        permitted: ({ can }) => can(Permission.LABEL_MANAGE),
      },
      {
        id: "quantifications",
        label: "Quantifications",
        description: "Measure and export the annotated data",
        icon: BarChart3,
        path: (datasetId) => `/dataset/${datasetId}/quantifications`,
        permitted: ({ can }) => can(Permission.EXPORT_QUANTIFICATION),
      },
    ],
  },
  {
    id: "models",
    label: "Artificial Intelligence",
    items: [
      {
        id: "model-zoo",
        label: "Model Zoo",
        description: "Browse and manage the available models",
        icon: Brain,
        // The zoo is a global page; it needs the dataset handed to it in router
        // state to scope its actions and to find its way back here.
        path: () => "/models",
        state: (datasetId) => ({ datasetId: Number(datasetId) }),
        permitted: ({ canAny }) =>
          canAny([Permission.AI_INTERACTIVE, Permission.AI_BATCH_INFER]),
      },
      {
        id: "model-orchestration",
        label: "Model Orchestration",
        description: "Choose default models and per-label routing",
        icon: Cpu,
        path: (datasetId) => `/dataset/${datasetId}/model-orchestration`,
        permitted: ({ canAny }) =>
          canAny([Permission.AI_INTERACTIVE, Permission.AI_BATCH_INFER]),
      },
      {
        id: "training",
        label: "Model Training",
        description: "Train a model on this dataset",
        icon: GraduationCap,
        path: (datasetId) => `/dataset/${datasetId}/training`,
        permitted: ({ can }) => can(Permission.AI_TRAIN),
      },
      {
        id: "inference",
        label: "Batch Inference",
        description: "Let your models annotate the whole dataset",
        icon: Wand2,
        path: (datasetId) => `/dataset/${datasetId}/inference`,
        permitted: ({ can }) => can(Permission.AI_BATCH_INFER),
      },
    ],
  },
  {
    id: "configurations",
    label: "Configuration",
    items: [
      {
        id: "access",
        label: "Manage Access",
        description: "Who can see, annotate and review this dataset",
        icon: Users2,
        path: (datasetId) => `/dataset/${datasetId}/access`,
        permitted: ({ canAny }) =>
          canAny([
            Permission.MEMBER_LIST,
            Permission.MEMBER_GRANT,
            Permission.INVITE_CREATE,
          ]),
      },
    ],
  },
];

/**
 * Is `item` the page currently open?
 *
 * Paths are compared exactly, so `/datamanagement/images` does not also light up
 * whatever else lives under `/datamanagement`. Pages that take a trailing id —
 * the editor and the read-only viewer — also match their sub-paths. Calibrating
 * is the annotation route with `?mode=calibrate`, so those two split on the
 * query.
 */
export const isItemActive = (item, datasetId, location) => {
  const [targetPath, targetQuery] = item.path(datasetId).split("?");
  const samePath =
    location.pathname === targetPath ||
    location.pathname.startsWith(`${targetPath}/`);
  if (!samePath) return false;
  const calibrating = (location.search || "").includes("mode=calibrate");
  if (targetQuery?.includes("mode=calibrate")) return calibrating;
  if (targetPath.endsWith("/annotate")) return !calibrating;
  return true;
};

/**
 * The dataset's section navigation, shown in the top bar.
 *
 * The overview itself is not a menu here: the dataset's name in the bar beside
 * this is the link to it, and two controls for one destination is one too many.
 *
 * Destinations the current role cannot use are left out entirely, the same way
 * the overview cards leave them out: a reviewer cannot grant themselves upload
 * rights, so a greyed-out "Data Management" entry would only be noise.
 * Everything is still re-checked server-side.
 */
const DatasetNav = ({ dataset, datasetId }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const permissions = usePermissions(dataset);
  const [openGroup, setOpenGroup] = useState(null);
  const containerRef = useRef(null);

  // A menu left open behind a click elsewhere would hang over the page below it.
  useEffect(() => {
    if (!openGroup) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!containerRef.current?.contains(event.target)) setOpenGroup(null);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpenGroup(null);
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [openGroup]);

  const id = datasetId ?? dataset?.id;
  if (!id) return null;

  // A group whose every destination is out of reach for this role is dropped
  // rather than opening onto an empty menu.
  const groups = NAV_GROUPS.map((group) => ({
    ...group,
    items: group.items.filter((item) => item.permitted(permissions)),
  })).filter((group) => group.items.length > 0);

  const go = (item) => {
    setOpenGroup(null);
    const state = item.state?.(id);
    navigate(item.path(id), state ? { state } : undefined);
  };

  const buttonPadding = "px-3 py-1.5 text-sm";

  return (
    <nav
      ref={containerRef}
      aria-label="Dataset sections"
      // Wraps rather than scrolls: an `overflow-x` here would clip the menus,
      // which hang below the bar.
      className="flex flex-wrap items-center gap-1"
    >
      {groups.map((group) => {
        const open = openGroup === group.id;
        const groupActive = group.items.some((item) =>
          isItemActive(item, id, location)
        );
        return (
          <div key={group.id} className="relative">
            <button
              onClick={() => setOpenGroup(open ? null : group.id)}
              aria-expanded={open}
              aria-haspopup="menu"
              className={`inline-flex items-center gap-1 rounded-lg font-medium transition-colors ${buttonPadding} ${
                groupActive
                  ? "bg-acS text-ac"
                  : open
                  ? "bg-hv text-t1"
                  : "text-t2 hover:bg-hv hover:text-t1"
              }`}
            >
              <span>{group.label}</span>
              <ChevronDown
                className={`w-3.5 h-3.5 transition-transform ${
                  open ? "rotate-180" : ""
                }`}
              />
            </button>

            {open && (
              <div
                role="menu"
                className="absolute left-0 mt-1 w-72 p-1.5 rounded-12 border border-ln bg-p1 shadow-lg z-50"
              >
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const active = isItemActive(item, id, location);
                  return (
                    <button
                      key={item.id}
                      role="menuitem"
                      onClick={() => go(item)}
                      aria-current={active ? "page" : undefined}
                      className={`w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-left transition-colors ${
                        active ? "bg-acS" : "hover:bg-hv"
                      }`}
                    >
                      <Icon
                        className={`w-4 h-4 mt-0.5 shrink-0 ${
                          active ? "text-ac" : "text-t3"
                        }`}
                      />
                      <span className="min-w-0">
                        <span
                          className={`block text-sm font-medium ${
                            active ? "text-ac" : "text-t1"
                          }`}
                        >
                          {item.label}
                        </span>
                        <span className="block text-xs text-t3">
                          {item.description}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </nav>
  );
};

export default DatasetNav;
