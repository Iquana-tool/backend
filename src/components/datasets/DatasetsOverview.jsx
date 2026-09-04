import React, { useState, useEffect } from "react";
import { useDataset } from "../../contexts/DatasetContext";
import { Plus, FolderOpen, User, UserCog } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../contexts/AuthContext";
import { usePermissions } from "../../hooks/usePermissions";
import { GLOBAL_ROLE_LABELS } from "../../utils/permissions";
import AuthButtons from "../auth/AuthButtons";
import DocsLink from "../ui/DocsLink";
import ReportBugLink from "../ui/ReportBugLink";
import AddDatasetModal from "./AddDatasetModal";
import UploadingModal from "./UploadingDatasetModal"
import CreateLabelsModal from "./CreateLabelsModal";
import DeleteDatasetModal from "./DeleteDatasetModal";
import DatasetCard from "./DatasetCard";
import { useDeleteDataset } from "../../hooks/useDeleteDataset";
import * as api from "../../api";
import { extractLabelsFromResponse } from "../../utils/labelHierarchy";
import { emptyPhaseCounts } from "../../utils/imageStatus";
import ThemeToggle from "../ui/ThemeToggle";
import Wordmark from '../Wordmark';

const DatasetsOverview = ({ onOpenDataset }) => {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuth();
  // Global capabilities: creating datasets and administering accounts are decided
  // by the account's platform role, not by any one dataset.
  const { canCreateDatasets, canManageUsers, globalRole } = usePermissions();
  const {
    datasets,
    loading,
    error,
    selectDataset,
    getAnnotationProgress,
    getSampleImages,
    fetchDatasets,
  } = useDataset();
  const [showAddModal, setShowAddModal] = useState(false);
  const [showLabelsModal, setShowLabelsModal] = useState(false);
  const [selectedDatasetForLabels, setSelectedDatasetForLabels] = useState(null);
  const [datasetImages, setDatasetImages] = useState({});
  const [datasetStats, setDatasetStats] = useState({});
  const [loadingData, setLoadingData] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [uploadingDatasetInfo, setUploadingDatasetInfo] = useState({
    title: "",
    description: "",
    total: 0
  });
  const [uploadingProgress, setUploadingProgress] = useState(0);
  useEffect(()=>{
    console.log(datasetStats)
  }, [datasetStats]);

  // Use the delete functionality hook
  const {
    showDeleteModal,
    datasetToDelete,
    isDeleting,
    initiateDelete,
    confirmDelete,
    cancelDelete,
  } = useDeleteDataset();
  useEffect(() => {
    console.log("Is Creating", isCreating);
  }, [isCreating]);
  // Fetch sample images immediately so cards display thumbnails without waiting for
  // progress calculations, and load annotation progress stats in the background.
  useEffect(() => {
    if (datasets.length === 0) {
      setLoadingData(false);
      return;
    }

    let isCancelled = false;

    // 1. Fetch sample thumbnails immediately for all datasets
    const thumbnailPromises = datasets.map(async (dataset) => {
      try {
        const images = await getSampleImages(dataset.id, 4);
        if (!isCancelled) {
          setDatasetImages((prev) => ({ ...prev, [dataset.id]: images }));
        }
      } catch (err) {
        console.error(`Error fetching images for dataset ${dataset.id}:`, err);
        if (!isCancelled) {
          setDatasetImages((prev) => ({ ...prev, [dataset.id]: [] }));
        }
      }
    });

    // 2. Fetch annotation progress in the background progressively
    setLoadingData(true);
    const progressPromises = datasets.map(async (dataset) => {
      try {
        const stats = await getAnnotationProgress(dataset.id);
        if (!isCancelled) {
          setDatasetStats((prev) => ({ ...prev, [dataset.id]: stats }));
        }
      } catch (err) {
        console.error(`Error fetching stats for dataset ${dataset.id}:`, err);
        if (!isCancelled) {
          setDatasetStats((prev) => ({
            ...prev,
            [dataset.id]: { ...emptyPhaseCounts(), total: 0 },
          }));
        }
      }
    });

    Promise.allSettled(thumbnailPromises);

    Promise.allSettled(progressPromises).finally(() => {
      if (!isCancelled) {
        setLoadingData(false);
      }
    });

    return () => {
      isCancelled = true;
    };
  }, [datasets, getSampleImages, getAnnotationProgress]);

  const handleOpenDataset = async (dataset) => {
    try {
      // Check if the dataset has any labels
      const labelsResponse = await api.fetchLabels(dataset.id);
      const labels = extractLabelsFromResponse(labelsResponse);

      // Filter out any invalid labels and orphaned sublabels
      const validLabels = labels.filter(label => {
        // Basic validation
        if (!label || typeof label !== 'object' || !label.id || !label.name || label.name.trim() === '') {
          return false;
        }

        // If this is a sublabel (has parent_id), check if its parent exists
        if (label.parent_id) {
          const parentExists = labels.some(l => l.id === label.parent_id);
          if (!parentExists) {
            console.warn(`Orphaned sublabel found: ${label.name} (parent ID ${label.parent_id} missing)`);
            return false; // Filter out orphaned sublabels
          }
        }

        return true;
      });

      if (!validLabels || validLabels.length === 0) {
        // No valid labels found, show the label creation modal
        setSelectedDatasetForLabels(dataset);
        setShowLabelsModal(true);
      } else {
        // Dataset has valid labels, proceed to open normally
        selectDataset(dataset);
        if (onOpenDataset) {
          onOpenDataset(dataset);
        }
      }
    } catch (error) {
      console.error('Error checking labels for dataset:', error);
      // On error, show the label creation modal as a fallback
      setSelectedDatasetForLabels(dataset);
      setShowLabelsModal(true);
    }
  };

  const handleLabelsCreated = () => {
    // After labels are created, select the dataset and open it
    if (selectedDatasetForLabels) {
      selectDataset(selectedDatasetForLabels);
      if (onOpenDataset) {
        onOpenDataset(selectedDatasetForLabels);
      }
    }
    setSelectedDatasetForLabels(null);
  };

  const handleLabelsModalClose = () => {
    setShowLabelsModal(false);
    setSelectedDatasetForLabels(null);
  };



  if (loading) {
    return (
      <div className="min-h-screen bg-well flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-acLn mx-auto mb-4"></div>
          <p className="text-t2">Loading datasets...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-well">
      {/* Header. Neutral panel, matching the shared Navbar and every other page
          header — not a solid accent fill, which fought the neutral text
          tokens these nav items use and made them nearly invisible. */}
      <div className="bg-p1 border-b border-ln">
        <div className="max-w-[98%] mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1
                className="text-2xl font-semibold tracking-tight text-t1 cursor-pointer hover:text-ac transition-colors duration-150"
                onClick={() => navigate('/')}
              >
                <Wordmark />
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              {isAuthenticated && user && (
                <div
                  className="flex items-center space-x-2 px-3 py-1.5 text-sm text-t3"
                  title={GLOBAL_ROLE_LABELS[globalRole]?.description}
                >
                  <User className="w-4 h-4" />
                  <span className="font-medium text-t2">{user.username}</span>
                  <span className="px-2 py-0.5 rounded-full bg-hv text-xs text-t2">
                    {GLOBAL_ROLE_LABELS[globalRole]?.label || globalRole}
                  </span>
                </div>
              )}
              {canManageUsers && (
                <button
                  onClick={() => navigate("/admin")}
                  title="Accounts and instance settings"
                  className="flex items-center space-x-2 bg-hv hover:bg-hv2 text-t2 hover:text-t1 py-2 px-4 rounded-lg transition-colors duration-150"
                >
                  <UserCog className="w-4 h-4" />
                  <span>Admin</span>
                </button>
              )}
              <DocsLink className="flex items-center space-x-2 bg-hv hover:bg-hv2 text-t2 hover:text-t1 py-2 px-4 rounded-lg transition-colors duration-150" />
              <ThemeToggle />
          <ReportBugLink className="flex items-center space-x-2 bg-hv hover:bg-hv2 text-t2 hover:text-t1 py-2 px-4 rounded-lg transition-colors duration-150" />
              <AuthButtons
                textColor="text-t2"
                buttonClass="flex items-center space-x-2 bg-hv hover:bg-errBg text-t2 hover:text-err py-2 px-4 rounded-lg transition-colors duration-150"
                showLogoutOnly={true}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-[98%] mx-auto px-4 py-8">
        {/* Page Header */}
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-3xl font-bold text-t1">Datasets</h2>
          {canCreateDatasets && (
            <button
              onClick={() => setShowAddModal(true)}
              className="flex items-center space-x-2 bg-accent text-onAccent px-6 py-3 rounded-lg hover:brightness-110 transition-colors"
            >
              <Plus className="w-5 h-5" />
              <span>Add new dataset</span>
            </button>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-errBg border border-errLn rounded-lg">
            <p className="text-err">{error}</p>
          </div>
        )}

        {/* Loading indicator for dataset data */}
        {loadingData && (
          <div className="mb-6 p-4 bg-acS border border-acLn rounded-lg">
            <p className="text-ac">Loading dataset information...</p>
          </div>
        )}

        {/* Datasets Grid */}
        {datasets.length === 0 ? (
          <div className="text-center py-12">
            <FolderOpen className="w-16 h-16 text-t3 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-t1 mb-2">
              No datasets yet
            </h3>
            {canCreateDatasets ? (
              <>
                <p className="text-t2 mb-6">
                  Get started by creating your first dataset
                </p>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="bg-accent text-onAccent px-6 py-3 rounded-lg hover:brightness-110 transition-colors"
                >
                  Create your first dataset
                </button>
              </>
            ) : (
              // Guests can only work in datasets they were invited to, so telling
              // them to create one would just lead to a 403.
              <p className="text-t2 max-w-md mx-auto">
                Your account cannot create datasets. Ask a dataset owner to send you an
                invite link, and it will show up here once you accept it.
              </p>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {datasets.map((dataset) => {
              const stats = datasetStats[dataset.id] || { ...emptyPhaseCounts(), total: 0 };
              const sampleImages = datasetImages[dataset.id] || [];

              return (
                <DatasetCard
                  key={dataset.id}
                  dataset={dataset}
                  stats={stats}
                  sampleImages={sampleImages}
                  onDelete={initiateDelete}
                  onOpenDataset={handleOpenDataset}
                  onShareSuccess={fetchDatasets}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Add Dataset Modal */}
      {showAddModal && !isCreating && (
        <AddDatasetModal
          isOpen={showAddModal}
          isCreating={isCreating}
          setIsCreating={setIsCreating}
          setCurrentProgress={setUploadingProgress}
          setDataSetInfo={setUploadingDatasetInfo}
          onClose={() => setShowAddModal(false)}
        />
      )}
      {showAddModal && isCreating && (
        <UploadingModal
            onClose={() => {
              setShowAddModal(false);
              setIsCreating(false);
            }}
            currentProgress={uploadingProgress}
            datasetInfo={uploadingDatasetInfo}
        />
      )}

      {/* Create Labels Modal */}
      {showLabelsModal && selectedDatasetForLabels && (
        <CreateLabelsModal
          isOpen={showLabelsModal}
          onClose={handleLabelsModalClose}
          dataset={selectedDatasetForLabels}
          onLabelsCreated={handleLabelsCreated}
        />
      )}

      {/* Delete Confirmation Modal */}
      <DeleteDatasetModal
        isOpen={showDeleteModal}
        dataset={datasetToDelete}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
        isDeleting={isDeleting}
      />
    </div>
  );
};

export default DatasetsOverview;
