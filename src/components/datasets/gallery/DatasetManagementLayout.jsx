import React from "react";
import { useParams } from "react-router-dom";
import { useDataset } from "../../../contexts/DatasetContext";
import DatasetGalleryHeader from "./DatasetGalleryHeader";
import SmallScreenMessage from "./SmallScreenMessage";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";
import {
  useGalleryLoadingData,
  useGalleryError,
  useGalleryActions
} from "../../../stores/selectors";
import { useDatasetGalleryData } from "../../../hooks/useDatasetGalleryData";

/**
 * Shared layout component for all dataset management pages.
 *
 * The dataset overview used to be a persistent left sidebar here. It carried the
 * dataset name (already in the header), its description, a Start Annotation
 * button and the label list — none of which are worth a fixed 25rem column on
 * every page, and the label list had a page of its own anyway. Navigation now
 * runs through the top bar (see `DatasetNav`), so the pages get the full width.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children - Content to display in the main area
 * @param {string|number} [props.datasetId] - Optional dataset ID (if not in URL params)
 */
const DatasetManagementLayout = ({
  children,
  datasetId: propDatasetId,
}) => {
  const { datasetId: paramDatasetId } = useParams();
  const { loading } = useDataset();

  // Use prop datasetId if provided, otherwise use URL param
  const datasetId = propDatasetId || paramDatasetId;

  // Zustand store selectors
  const loadingData = useGalleryLoadingData();
  const error = useGalleryError();
  const galleryActions = useGalleryActions();

  // Use custom hook for data fetching and initialization
  const dataset = useDatasetGalleryData(datasetId, galleryActions);

  if (loading || loadingData) {
    return <LoadingState />;
  }

  if (error || !dataset) {
    return <ErrorState error={error} />;
  }

  return (
    <div className="min-h-screen bg-well">
      <SmallScreenMessage />

      {/* Large Screen Content - Show dataset management layout on screens 1024px and above */}
      <div className="hidden lg:flex lg:flex-col lg:h-screen">
        <DatasetGalleryHeader dataset={dataset} />

        {/* Main Content */}
        <div className="max-w-full mx-auto flex flex-1 min-h-0 w-full">
          <div className="flex-1 overflow-hidden">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetManagementLayout;
