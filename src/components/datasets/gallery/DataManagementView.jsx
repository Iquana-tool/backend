import React, { useCallback, useState } from "react";
import ImageGallery from "./ImageGallery";
import { usePermissions, Permission } from "../../../hooks/usePermissions";
import * as api from "../../../api";

const DataManagementView = ({ images, dataset, onImageClick, onImagesUpdated, onShowQuantifications }) => {
  // The selection itself lives in the gallery, next to the checkboxes and the
  // bulk bar; this view only owns the confirmation step and the delete calls.
  const [pendingDelete, setPendingDelete] = useState([]);
  const [isDeleting, setIsDeleting] = useState(false);
  const { can } = usePermissions(dataset);
  const canDelete = can(Permission.IMAGE_DELETE);

  const handleImageClick = (image) => {
    // Navigate to annotation page
    onImageClick(image);
  };

  const refreshImages = useCallback(() => {
    if (onImagesUpdated) {
      onImagesUpdated();
    } else {
      window.location.reload();
    }
  }, [onImagesUpdated]);

  const handleDeleteSelected = async () => {
    if (pendingDelete.length === 0) return;

    setIsDeleting(true);
    try {
      // One request per image, in parallel: a failure on one must not stop the
      // rest, so each rejection is caught and counted instead of thrown.
      const results = await Promise.all(
        pendingDelete.map((image) =>
          api.deleteImage(image.id).catch((err) => {
            console.error(`Failed to delete image ${image.id}:`, err);
            return { success: false, imageId: image.id };
          })
        )
      );

      const failed = results.filter((r) => !r?.success);
      if (failed.length > 0) {
        console.error("Some images failed to delete:", failed);
        alert(`${failed.length} of ${pendingDelete.length} images could not be deleted.`);
      }

      refreshImages();
      setPendingDelete([]);
    } catch (error) {
      console.error("Error deleting images:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteSingle = async (imageId, e) => {
    e.stopPropagation(); // Prevent navigation to annotation

    if (!window.confirm("Are you sure you want to delete this image?")) {
      return;
    }

    try {
      const result = await api.deleteImage(imageId);
      if (result.success) {
        refreshImages();
      } else {
        alert("Failed to delete image: " + (result.message || "Unknown error"));
      }
    } catch (error) {
      console.error("Error deleting image:", error);
      alert("Failed to delete image. Please try again.");
    }
  };

  return (
    <div className="h-full flex flex-col bg-p1">
      {/* Header */}
      <div className="p-3 sm:p-4 border-b border-ln bg-p1 sticky top-0 z-10">
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <h2 className="text-lg sm:text-xl font-bold text-t1">
            Data Management
          </h2>
        </div>
      </div>

      {/* Image Gallery */}
      <div className="flex-1 overflow-hidden">
        <ImageGallery
          images={images}
          onImageClick={handleImageClick}
          dataset={dataset}
          onDeleteImage={handleDeleteSingle}
          onImagesUpdated={onImagesUpdated}
          onBulkDelete={canDelete ? setPendingDelete : undefined}
          onShowQuantifications={onShowQuantifications}
        />
      </div>

      {/* Delete Confirmation Modal */}
      {pendingDelete.length > 0 && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div
              className="fixed inset-0 transition-opacity"
              onClick={() => !isDeleting && setPendingDelete([])}
            >
              <div className="absolute inset-0 bg-t3 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-p1 rounded-xl text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-p1 px-6 pt-6 pb-4">
                <h3 className="text-lg font-medium text-t1 mb-4">
                  Delete {pendingDelete.length} image{pendingDelete.length > 1 ? 's' : ''}?
                </h3>
                <p className="text-sm text-t2 mb-6">
                  This action cannot be undone. The selected image{pendingDelete.length > 1 ? 's' : ''} and
                  {pendingDelete.length > 1 ? ' their' : ' its'} annotations will be permanently deleted.
                </p>

                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => setPendingDelete([])}
                    disabled={isDeleting}
                    className="px-4 py-2 text-sm text-t2 bg-well rounded-lg hover:bg-hv2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleDeleteSelected}
                    disabled={isDeleting}
                    className="px-4 py-2 text-sm text-onAccent bg-err rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isDeleting ? "Deleting..." : "Delete"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataManagementView;
