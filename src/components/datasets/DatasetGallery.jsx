import React, { useCallback, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import { useDataset } from "../../contexts/DatasetContext";
import DataManagementView from "./gallery/DataManagementView";
import LabelManagementView from "./gallery/LabelManagementView";
import ManagementCardsView from "./gallery/ManagementCardsView";
import CocoExportModal from "./gallery/CocoExportModal";
import AnnotationQueueModal from "./gallery/AnnotationQueueModal";
import DatasetManagementLayout from "./gallery/DatasetManagementLayout";
import * as api from "../../api";
import { normalizeImage } from "../../hooks/useDatasetGalleryData";
import { usePermissions } from "../../hooks/usePermissions";
import { Permission } from "../../utils/permissions";
import { 
  useGalleryImages,
  useGalleryLabels,
  useGalleryActions 
} from "../../stores/selectors";

const DatasetGallery = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { currentDataset } = useDataset();
  
  // Zustand store selectors
  const images = useGalleryImages();
  const labels = useGalleryLabels();
  const galleryActions = useGalleryActions();

  const [showCocoModal, setShowCocoModal] = useState(false);
  const [showQueueModal, setShowQueueModal] = useState(false);

  const { can } = usePermissions(currentDataset);

  // Derive current view from URL path so refresh preserves the view
  const pathname = location.pathname;
  const currentView = pathname.endsWith("/images")
    ? "dataManagement"
    : pathname.endsWith("/labels")
    ? "labelManagement"
    : "cards";
  
  // Get dataset from context (set by DatasetManagementLayout)
  const dataset = currentDataset;

  // Handle labels updated from LabelManagementView
  const handleLabelsUpdated = useCallback((updatedLabels) => {
    galleryActions.setLabels(updatedLabels);
  }, [galleryActions]);

  // The Annotation card opens the queue builder rather than jumping straight into
  // the editor: the queue defines the order images are worked in.
  const handleAnnotationClick = () => setShowQueueModal(true);

  // Called by the queue modal once a queue is built or resumed. Land on the first
  // not-yet-finished image in queue order (fall back to the first image, then to
  // the general page), and hand the order over via router state so the editor's
  // loader can apply it without a reorder flash — it also re-reads the saved queue.
  const handleQueueStart = (imageIds) => {
    setShowQueueModal(false);
    if (!imageIds || imageIds.length === 0) {
      navigate(`/dataset/${dataset.id}/annotate`);
      return;
    }
    const statusById = new Map(images.map((img) => [img.id, img]));
    const firstUnfinished = imageIds.find((id) => {
      const img = statusById.get(id);
      return !img || (img.status !== "finished" && !img.finished);
    });
    const targetId = firstUnfinished ?? imageIds[0];
    navigate(`/dataset/${dataset.id}/annotate/${targetId}`, {
      state: { queueImageIds: imageIds },
    });
  };

  const handleImageClick = (image) => {
    navigate(`/dataset/${dataset.id}/annotate/${image.id}`);
  };

  // Calibrate opens the annotation workspace in its Calibrate tab rather than a
  // page of its own — calibrating means working on the image, at the same zoom and
  // pan as annotating it. `?mode=calibrate` is read once on arrival and then
  // stripped, so switching mode afterwards does not fight the URL.
  //
  // Lands on the first image with no calibration at all, so the card resumes where
  // the work stopped instead of always reopening image one.
  const handleCalibrateClick = async () => {
    if (!dataset) return;
    const open = (imageId) =>
      navigate(
        imageId
          ? `/dataset/${dataset.id}/annotate/${imageId}?mode=calibrate`
          : `/dataset/${dataset.id}/annotate?mode=calibrate`
      );
    try {
      const response = await api.fetchImagesWithAnnotationStatus(
        dataset.id, 'not_started', 'calibrate'
      );
      const pending = response?.success ? response.image_data || [] : [];
      open(pending[0]?.image_id);
    } catch (err) {
      // The lookup is a convenience; failing it should still open the tab.
      console.error('Error finding an uncalibrated image:', err);
      open(null);
    }
  };

  // Refresh images list - uses normalizeImage to ensure consistent shape
  // (the API returns image_id not id; without normalization data-image-id is
  // undefined and the IntersectionObserver cannot load thumbnails)
  const refreshImages = useCallback(async () => {
    if (!dataset) return;
    
    try {
      const imagesResponse = await api.fetchImages(dataset.id);
      if (imagesResponse.success) {
        const imageDataList = imagesResponse.image_data || imagesResponse.images || [];
        galleryActions.setImages(imageDataList.map(normalizeImage));
      }
    } catch (err) {
      console.error("Error refreshing images:", err);
    }
  }, [dataset, galleryActions]);

  // Card click handlers
  const handleDataManagementClick = () => {
    navigate(`/dataset/${datasetId}/datamanagement/images`);
  };

  const handleModelZooClick = () => {
    navigate("/models", { state: { datasetId: dataset?.id } });
  };

  const handleQuantificationsClick = () => {
    navigate(`/dataset/${datasetId}/quantifications`);
  };

  // The second entry point into per-image inspection (issue #15): from Data Management,
  // where the image itself is what the viewer is already looking at. The other one is a
  // surface on the dataset quantification page, for someone who arrived from the numbers.
  const handleImageQuantificationsClick = (image) => {
    navigate(`/dataset/${datasetId}/quantifications/image/${image.id}`);
  };

  const handleLabelManagementClick = () => {
    navigate(`/dataset/${datasetId}/datamanagement/labels`);
  };

  const handleModelTrainingClick = () => {
    navigate(`/dataset/${datasetId}/training`);
  };

  const handleModelOrchestrationClick = () => {
    navigate(`/dataset/${datasetId}/model-orchestration`);
  };

  const handleBatchInferenceClick = () => {
    navigate(`/dataset/${datasetId}/inference`);
  };

  const handleBrowseAnnotations = () => {
    navigate(`/dataset/${datasetId}/view`);
  };

  const handleManageAccessClick = () => {
    navigate(`/dataset/${datasetId}/access`);
  };

  const handleReviewClick = () => {
    navigate(`/dataset/${datasetId}/review`);
  };

  const handleCorrectClick = () => {
    navigate(`/dataset/${datasetId}/correct`);
  };

  return (
    <DatasetManagementLayout>
      <div className="h-full overflow-hidden">
        {currentView === "cards" ? (
          <ManagementCardsView
            dataset={dataset}
            onDataManagementClick={handleDataManagementClick}
            onModelZooClick={handleModelZooClick}
            onQuantificationsClick={handleQuantificationsClick}
            onAnnotationClick={handleAnnotationClick}
            onLabelManagementClick={handleLabelManagementClick}
            onExportCocoClick={() => setShowCocoModal(true)}
            onModelTrainingClick={handleModelTrainingClick}
            onModelOrchestrationClick={handleModelOrchestrationClick}
            onBatchInferenceClick={handleBatchInferenceClick}
            onBrowseAnnotations={handleBrowseAnnotations}
            onManageAccessClick={handleManageAccessClick}
            onCalibrateClick={handleCalibrateClick}
            onReviewClick={handleReviewClick}
            onCorrectClick={handleCorrectClick}
          />
        ) : currentView === "dataManagement" ? (
          <DataManagementView
            images={images}
            dataset={dataset}
            onImageClick={handleImageClick}
            onImagesUpdated={refreshImages}
            onShowQuantifications={
              can(Permission.EXPORT_QUANTIFICATION)
                ? handleImageQuantificationsClick
                : undefined
            }
          />
        ) : currentView === "labelManagement" ? (
          <LabelManagementView
            dataset={dataset}
            labels={labels}
            onLabelsUpdated={handleLabelsUpdated}
          />
        ) : null}
      </div>

      <CocoExportModal
        isOpen={showCocoModal}
        onClose={() => setShowCocoModal(false)}
        dataset={dataset}
      />

      <AnnotationQueueModal
        isOpen={showQueueModal}
        onClose={() => setShowQueueModal(false)}
        dataset={dataset}
        onStart={handleQueueStart}
      />
    </DatasetManagementLayout>
  );
};

export default DatasetGallery; 