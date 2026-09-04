import React, { useState, useEffect, useCallback, useMemo } from "react";
import { useNavigate, useLocation, useParams } from "react-router-dom";
import { User, Brain, Search, Star, ArrowLeft, Loader2, GraduationCap, ArrowUpDown } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { useToast } from "../contexts/ToastContext";
import AuthButtons from "../components/auth/AuthButtons";
import DocsLink from "../components/ui/DocsLink";
import ReportBugLink from "../components/ui/ReportBugLink";
import ModelChip from "../components/models/ModelChip";
import ModelDetailPanel from "../components/models/ModelDetailPanel";
import { SORT_STRATEGIES, DEFAULT_SORT, hasAnyPerfStats } from "../components/models/modelStats";
import TrainingModal from "../components/models/TrainingModal";
import TrainingJobCard from "../components/models/TrainingJobCard";
import DatasetManagementLayout from "../components/datasets/gallery/DatasetManagementLayout";
import {
  getAllModels,
  startSemanticTraining,
  getSemanticTrainingStatus,
  cancelSemanticTraining,
} from "../api/training";
import { getModelFavorites, setModelFavorite, clearModelFavorite } from "../api/models";
import { TASK_ORDER, getTaskMeta } from "../constants/tasks";
import ThemeToggle from "../components/ui/ThemeToggle";
import Wordmark from '../components/Wordmark';

// The toolbox serves `tags` as a dict (e.g. { domain: "general" }), but older
// payloads may use an array or comma string. Normalize all three into
// [{ key?, value }] so the card can render them.
const normalizeTagEntries = (tags) => {
  if (!tags) return [];
  if (Array.isArray(tags)) {
    return tags.map((t) => ({ value: String(t) })).filter((t) => t.value);
  }
  if (typeof tags === "string") {
    return tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean)
      .map((value) => ({ value }));
  }
  if (typeof tags === "object") {
    return Object.entries(tags)
      .filter(([, value]) => value != null && String(value) !== "")
      .map(([key, value]) => ({ key, value: String(value) }));
  }
  return [];
};

const getTagValue = (tags, key) => {
  if (!tags) return null;
  if (Array.isArray(tags)) {
    const entry = tags.find((tag) => tag?.key === key);
    return entry?.value ?? null;
  }
  return typeof tags === "object" ? tags[key] ?? null : null;
};

const parseTagList = (value) => {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
};

// Transform a merged backend model into the card's UI shape. `tasks` is the
// array of task keys the model serves (already merged in getAllModels).
const transformModel = (model) => ({
  identifier: model.identifier || model.registry_key,
  name: model.name,
  description: model.description,
  tags: normalizeTagEntries(model.tags),
  tasks: Array.isArray(model.tasks) ? model.tasks : [],
  badges: Array.isArray(model.badges) ? model.badges : [],
  usageTip: model.usage_tip,
  infoUrl: model.info_url,
  status: model.status,
  promptTypesSupported: Array.isArray(model.prompt_types_supported)
    ? model.prompt_types_supported
    : [],
  refinementSupported: model.refinement_supported === true,
  labelId: model.label_id,
  labelIds: Array.isArray(model.label_ids)
    ? model.label_ids
    : model.label_id != null
    ? [model.label_id]
    : [],
  predictedLabelNames: parseTagList(getTagValue(model.tags, "trained_label_names")),
  trainedOnDatasetId: getTagValue(model.tags, "trained_on_dataset_id")
    || getTagValue(model.tags, "dataset_id"),
  trainedOnDatasetName: getTagValue(model.tags, "trained_on_dataset_name")
    || getTagValue(model.tags, "dataset_name"),
  trainable: model.trainable === true,
  pretrained: model.pretrained !== false,
  architecture: model.architecture || null,
  license: model.license || null,
  inputResolution: Array.isArray(model.input_resolution) ? model.input_resolution : null,
  performance: model.performance && typeof model.performance === "object" ? model.performance : null,
});

const POLL_INTERVAL_MS = 4000;

const normalizeFavorites = (result) =>
  result?.success && result?.favorites && typeof result.favorites === "object"
    ? result.favorites
    : {};

const ModelZooPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { datasetId } = useParams();
  const { isAuthenticated, user } = useAuth();
  const { addToast } = useToast();

  const [models, setModels] = useState([]);
  const [favorites, setFavorites] = useState({}); // { task: registry_key }
  const [selectedTask, setSelectedTask] = useState("all"); // "all" | task key
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState(DEFAULT_SORT);
  const [selectedModelId, setSelectedModelId] = useState(null); // detail-panel selection
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Training modal state (semantic segmentation; instance seg has its own page)
  const [trainingModal, setTrainingModal] = useState({ isOpen: false, model: null, actionType: null });
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [cancellingTaskId, setCancellingTaskId] = useState(null);

  const datasetIdFromState = location.state?.datasetId || datasetId;
  const isFromDatasetManagement = !!datasetIdFromState;

  const resyncFavorites = useCallback(async () => {
    try {
      const result = await getModelFavorites();
      if (!result?.success) return false;
      setFavorites(normalizeFavorites(result));
      return true;
    } catch (_) {
      return false;
    }
  }, []);

  const refetchModels = useCallback(async () => {
    try {
      const result = await getAllModels();
      if (result.success && result.models) {
        setModels(result.models.map(transformModel));
      }
    } catch (_) {
      // ignore
    }
  }, []);

  // Initial load: models + favorites in parallel.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setIsLoading(true);
      setError(null);
      const [modelsResult, favoritesResult] = await Promise.allSettled([
        getAllModels(),
        getModelFavorites(),
      ]);
      if (cancelled) return;

      if (modelsResult.status === "fulfilled") {
        const result = modelsResult.value;
        if (result?.success && result.models) {
          setModels(result.models.map(transformModel));
        } else {
          setError(result?.error || "Failed to load models from backend");
        }
      } else {
        setError(modelsResult.reason?.message || "Failed to load models");
      }

      if (favoritesResult.status === "fulfilled") {
        setFavorites(normalizeFavorites(favoritesResult.value));
      } else {
        setFavorites({});
      }
      setIsLoading(false);
    };
    load();
    return () => { cancelled = true; };
  }, []);

  // Poll training job status for active jobs
  useEffect(() => {
    const active = trainingJobs.filter((j) => j.status === "PENDING" || j.status === "STARTED");
    if (active.length === 0) return;

    const poll = async () => {
      for (const job of active) {
        try {
          const res = await getSemanticTrainingStatus(job.task_id);
          const status = res?.result?.status?.toUpperCase?.() ?? job.status;
          const progress = res?.result?.progress ?? res?.result?.info ?? null;
          const failureMessage =
            status === "FAILURE"
              ? typeof progress === "string"
                ? progress
                : progress?.message ?? progress?.error ?? "Training failed."
              : null;

          setTrainingJobs((prev) =>
            prev.map((j) =>
              j.task_id !== job.task_id
                ? j
                : { ...j, status, progress: progress ?? j.progress, error: failureMessage ?? j.error }
            )
          );
          if (status === "SUCCESS") refetchModels();
        } catch (_) {
          // keep current state on poll error
        }
      }
    };

    const t = setInterval(poll, POLL_INTERVAL_MS);
    poll();
    return () => clearInterval(t);
  }, [trainingJobs, refetchModels]);

  const isFavorite = useCallback(
    (model) => {
      if (selectedTask !== "all") {
        return favorites[selectedTask] === model.identifier;
      }
      return (model.tasks || []).some((task) => favorites[task] === model.identifier);
    },
    [favorites, selectedTask]
  );

  // Star toggles the model as personal favorite for a specific task.
  const handleToggleFavorite = useCallback(
    async (model, specificTask = null) => {
      const id = model.identifier;
      const targetTask =
        specificTask ||
        (selectedTask !== "all"
          ? selectedTask
          : model.tasks?.length === 1
          ? model.tasks[0]
          : null);

      if (!targetTask) return;

      const wasFavorite = favorites[targetTask] === id;

      setFavorites((prev) => {
        const next = { ...prev };
        if (wasFavorite) {
          if (next[targetTask] === id) delete next[targetTask];
        } else {
          next[targetTask] = id;
        }
        return next;
      });

      let result;
      try {
        result = wasFavorite
          ? await clearModelFavorite(targetTask)
          : await setModelFavorite(targetTask, id);
      } catch (_) {
        // Treat rejected API calls the same as an explicit failure response.
      }

      if (!result?.success) {
        const resynced = await resyncFavorites();
        if (!resynced) setFavorites(favorites);
        addToast({ message: "Couldn't update favorites. Try again.", type: "error" });
      }
    },
    [favorites, selectedTask, addToast, resyncFavorites]
  );

  const handleModelAction = (model, actionType) => {
    if (actionType !== "finetuning" && actionType !== "training") return;

    // Instance-segmentation training is dataset-scoped (needs class/label
    // selection), so it lives on its own page.
    if ((model.tasks || []).includes("instance-segmentation")) {
      if (datasetIdFromState) {
        navigate(`/dataset/${datasetIdFromState}/training`, { state: { modelKey: model.identifier } });
      } else {
        addToast({
          message: "Open the Model Zoo from a dataset to fine-tune instance segmentation models.",
          type: "info",
        });
      }
      return;
    }

    addToast({ message: `${model.name} is not fine-tunable here.`, type: "info" });
  };

  const handleTrainingSubmit = async (trainingParams) => {
    const response = await startSemanticTraining(trainingParams);
    if (!response?.success) throw new Error(response?.message || "Training failed");
    const taskId = response?.result?.task_id;
    if (!taskId) throw new Error("Server did not return a task ID.");
    const initialState = (response?.result?.state || "PENDING").toUpperCase();
    setTrainingJobs((prev) => [
      ...prev,
      {
        task_id: taskId,
        model_key: trainingModal.model?.identifier,
        model_name: trainingModal.model?.name,
        dataset_id: trainingParams.dataset_id,
        status: initialState === "STARTED" ? "STARTED" : "PENDING",
        progress: response?.result?.data ?? null,
        error: null,
        startedAt: new Date().toISOString(),
      },
    ]);
    addToast({ message: "Training started. Track progress in the “Training runs” section below.", type: "success" });
  };

  const handleCancelTraining = async (taskId) => {
    setCancellingTaskId(taskId);
    try {
      await cancelSemanticTraining(taskId);
      setTrainingJobs((prev) => prev.map((j) => (j.task_id === taskId ? { ...j, status: "REVOKED" } : j)));
      addToast({ message: "Training cancelled.", type: "success" });
    } catch (err) {
      addToast({ message: err?.message || "Failed to cancel training", type: "error" });
    } finally {
      setCancellingTaskId(null);
    }
  };

  const handleBack = () => {
    if (datasetIdFromState) navigate(`/dataset/${datasetIdFromState}/datamanagement`);
    else if (window.history.length > 1) navigate(-1);
    else navigate("/datasets");
  };

  // Task facets that actually have models, in canonical order.
  const presentTasks = useMemo(
    () => TASK_ORDER.filter((t) => models.some((m) => (m.tasks || []).includes(t))),
    [models]
  );

  const filteredModels = useMemo(() => {
    const q = search.trim().toLowerCase();
    return models.filter((m) => {
      if (favoritesOnly && !isFavorite(m)) return false;
      if (selectedTask !== "all" && !(m.tasks || []).includes(selectedTask)) return false;
      if (q) {
        const haystack = `${m.name || ""} ${m.identifier || ""} ${m.description || ""}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });
  }, [models, favoritesOnly, selectedTask, search, isFavorite]);

  // Sorting is client-side: the zoo already fetches every model, so ordering by
  // name / inference time / params / GFLOPs is a pure re-sort of the filtered set.
  const showPerfSort = useMemo(() => hasAnyPerfStats(models), [models]);
  const sortOptions = useMemo(
    () => (showPerfSort ? Object.keys(SORT_STRATEGIES) : ["name"]),
    [showPerfSort]
  );
  const sortedModels = useMemo(() => {
    const strategy = SORT_STRATEGIES[sortBy] || SORT_STRATEGIES[DEFAULT_SORT];
    return [...filteredModels].sort(strategy.compare);
  }, [filteredModels, sortBy]);

  const favoriteCount = useMemo(() => models.filter(isFavorite).length, [models, isFavorite]);

  // Keep a valid model selected for the detail panel as filters/sorting change.
  useEffect(() => {
    if (sortedModels.length === 0) {
      if (selectedModelId !== null) setSelectedModelId(null);
      return;
    }
    if (!sortedModels.some((m) => m.identifier === selectedModelId)) {
      setSelectedModelId(sortedModels[0].identifier);
    }
  }, [sortedModels, selectedModelId]);

  const selectedModel = useMemo(
    () => sortedModels.find((m) => m.identifier === selectedModelId) || null,
    [sortedModels, selectedModelId]
  );

  // If perf stats disappear (e.g. only unsorted models remain), fall back to name.
  useEffect(() => {
    if (!sortOptions.includes(sortBy)) setSortBy(DEFAULT_SORT);
  }, [sortOptions, sortBy]);

  const FacetButton = ({ active, onClick, icon: Icon, children }) => (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-sm font-medium transition-colors border ${
        active
          ? "bg-accent text-onAccent border-acLn"
          : "bg-p1 text-t2 border-ln hover:border-ln2 hover:text-t1"
      }`}
    >
      {Icon && <Icon className="w-4 h-4" />}
      {children}
    </button>
  );

  const ModelZooContent = () => (
    <div className="h-full flex flex-col bg-p1">
      {isFromDatasetManagement && datasetIdFromState && (
        <div className="p-3 sm:p-4 border-b border-ln bg-p1 sticky top-0 z-10">
          <div className="flex items-center space-x-2 sm:space-x-3 lg:space-x-4">
            <button
              onClick={() => navigate(`/dataset/${datasetIdFromState}/datamanagement`)}
              className="flex items-center space-x-1.5 sm:space-x-2 text-t2 hover:text-t1 transition-colors text-sm sm:text-base"
            >
              <ArrowLeft size={18} className="sm:w-5 sm:h-5" />
              <span className="hidden sm:inline">Back to Overview</span>
              <span className="sm:hidden">Back</span>
            </button>
            <div className="h-5 sm:h-6 w-px bg-ln2"></div>
            <h2 className="text-lg sm:text-xl font-bold text-t1">Model Zoo</h2>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 bg-well">
        <div className="max-w-[98%] w-full h-full mx-auto px-4 py-5 flex flex-col min-h-0">
          {isLoading && (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="w-12 h-12 text-ac animate-spin mb-4" />
              <p className="text-t2">Loading models...</p>
            </div>
          )}

          {error && !isLoading && (
            <div className="bg-errBg border border-errLn rounded-lg p-6 text-center">
              <p className="text-err font-medium mb-2">Failed to load models</p>
              <p className="text-err text-sm">{error}</p>
            </div>
          )}

          {!isLoading && !error && (
            <div className="flex-1 min-h-0 flex flex-col">
              {!isFromDatasetManagement && (
                <div className="mb-5 shrink-0">
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 bg-acS rounded-8 flex items-center justify-center">
                      <Brain className="w-6 h-6 text-ac" />
                    </div>
                    <div>
                      <h2 className="text-3xl font-bold text-t1">Model Zoo</h2>
                      <p className="text-t2 mt-1">
                        Explore models and star your defaults for each task
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Training runs */}
              {trainingJobs.length > 0 && (
                <div className="mb-5 shrink-0">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="h-1 w-12 bg-accent rounded-full" />
                    <h3 className="text-lg font-bold text-t1 flex items-center gap-2">
                      <GraduationCap className="w-5 h-5 text-ac" />
                      Training runs
                    </h3>
                    <div className="h-1 flex-1 bg-ln2 rounded-full" />
                  </div>
                  <div className="space-y-3 max-h-[28vh] overflow-y-auto pr-1">
                    {trainingJobs.map((job) => (
                      <TrainingJobCard
                        key={job.task_id}
                        job={job}
                        onCancel={handleCancelTraining}
                        isCancelling={cancellingTaskId === job.task_id}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Master–detail: viewer on the left, model preview list on the right */}
              <div className="flex-1 min-h-0 flex gap-5">
                {/* Detail viewer */}
                <div className="flex-1 min-w-0 bg-p1 rounded-2xl border border-ln shadow-sm overflow-hidden">
                  <ModelDetailPanel
                    model={selectedModel}
                    isFavorite={selectedModel ? isFavorite(selectedModel) : false}
                    selectedTask={selectedTask}
                    favorites={favorites}
                    onToggleFavorite={handleToggleFavorite}
                    onAction={handleModelAction}
                  />
                </div>

                {/* Preview list + controls */}
                <div className="w-[340px] shrink-0 flex flex-col min-h-0">
                  <div className="relative mb-2.5 shrink-0">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-t3" />
                    <input
                      type="text"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search models"
                      className="w-full pl-9 pr-3 py-1.5 text-sm bg-p1 border border-ln rounded-full focus:outline-none focus:ring-1 focus:ring-ac focus:border-ac"
                    />
                  </div>

                  {sortOptions.length > 1 && (
                    <div className="relative mb-2.5 shrink-0">
                      <ArrowUpDown className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-t3 pointer-events-none" />
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value)}
                        aria-label="Sort models"
                        className="w-full pl-9 pr-8 py-1.5 text-sm bg-p1 border border-ln rounded-full appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-ac focus:border-ac"
                      >
                        {sortOptions.map((key) => (
                          <option key={key} value={key}>
                            Sort: {SORT_STRATEGIES[key].label}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-1.5 mb-3 shrink-0">
                    <FacetButton active={selectedTask === "all"} onClick={() => setSelectedTask("all")}>
                      All
                    </FacetButton>
                    {presentTasks.map((task) => {
                      const meta = getTaskMeta(task);
                      return (
                        <FacetButton
                          key={task}
                          active={selectedTask === task}
                          onClick={() => setSelectedTask(task)}
                        >
                          {meta.short}
                        </FacetButton>
                      );
                    })}
                    {favoriteCount > 0 && (
                      <FacetButton
                        active={favoritesOnly}
                        onClick={() => setFavoritesOnly((v) => !v)}
                        icon={Star}
                      >
                        Favorites
                      </FacetButton>
                    )}
                  </div>

                  <div className="flex-1 min-h-0 overflow-y-auto -mr-1 pr-1 space-y-2">
                    {sortedModels.length > 0 ? (
                      sortedModels.map((model) => (
                        <ModelChip
                          key={model.identifier}
                          model={model}
                          isSelected={model.identifier === selectedModelId}
                          isFavorite={isFavorite(model)}
                          selectedTask={selectedTask}
                          favorites={favorites}
                          onSelect={(m) => setSelectedModelId(m.identifier)}
                          onToggleFavorite={handleToggleFavorite}
                        />
                      ))
                    ) : (
                      <div className="text-center py-10 px-4">
                        <Brain className="w-12 h-12 text-t3 mx-auto mb-3" />
                        <h3 className="text-sm font-semibold text-t1 mb-1">No models found</h3>
                        <p className="text-xs text-t3">
                          {models.length === 0
                            ? "No models are registered yet."
                            : "Try a different filter or search."}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <TrainingModal
        isOpen={trainingModal.isOpen}
        onClose={() => setTrainingModal({ isOpen: false, model: null, actionType: null })}
        model={trainingModal.model}
        onSubmit={handleTrainingSubmit}
        datasetId={datasetIdFromState}
      />
    </div>
  );

  if (isFromDatasetManagement && datasetIdFromState) {
    return (
      <DatasetManagementLayout datasetId={datasetIdFromState}>
        {ModelZooContent()}
      </DatasetManagementLayout>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-well">
      <div className="bg-p1 text-t1 border-b border-ln shrink-0">
        <div className="max-w-[98%] mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button onClick={handleBack} className="flex items-center gap-[7px] text-t2 hover:text-ac transition-colors duration-150">
                <ArrowLeft size={20} />
                <span>Back</span>
              </button>
              <div className="h-6 w-px bg-ln"></div>
              <h1
                className="text-2xl font-semibold tracking-tight cursor-pointer hover:text-ac transition-colors duration-150"
                onClick={() => navigate("/")}
              >
                <Wordmark />
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              {isAuthenticated && user && (
                <div className="flex items-center gap-[6px] px-3 py-1.5 text-sm text-t3">
                  <User className="w-4 h-4" />
                  <span className="font-medium">{user.username}</span>
                </div>
              )}
              <DocsLink className="flex items-center gap-[7px] bg-hv hover:bg-hv2 text-t2 hover:text-t1 py-2 px-4 rounded-6 transition-colors duration-150" />
              <ThemeToggle />
          <ReportBugLink />
              <AuthButtons showLogoutOnly={true} />
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        {ModelZooContent()}
      </div>
    </div>
  );
};

export default ModelZooPage;
