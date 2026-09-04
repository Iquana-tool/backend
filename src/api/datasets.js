import { handleApiError, getAuthHeaders, buildUrl } from "../api/util";
import { getImageById } from "./images"; // Import the image fetching function

import { API_BASE_URL } from "./config";

// Dataset API functions
export const fetchDatasets = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/datasets/all`, {
            headers: getAuthHeaders(),
        });
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

export const createDataset = async (name, description, datasetType) => {
    try {
        const url = buildUrl(API_BASE_URL, '/datasets/create', {
            name: name,
            description: description,
            dataset_type: datasetType
        });

        const response = await fetch(url, {
            method: "POST",
            headers: getAuthHeaders(),
        });
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

export const deleteDataset = async (datasetId) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/datasets/${datasetId}`,
            {
                method: "DELETE",
                headers: getAuthHeaders(),
            }
        );
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

export const shareDataset = async (datasetId, shareWithUsername) => {
    try {
        const url = buildUrl(API_BASE_URL, `/datasets/${datasetId}/share`, {
            share_with_username: shareWithUsername,
        });
        const response = await fetch(url, {
            method: "POST",
            headers: getAuthHeaders(),
        });
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

export const getDataset = async (datasetId) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/datasets/${datasetId}`,
            {
                headers: getAuthHeaders(),
            }
        );
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

// Get annotation progress for a dataset
export const getAnnotationProgress = async (datasetId) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/datasets/${datasetId}/progress`,
            {
                headers: getAuthHeaders(),
            }
        );
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

// Get sample images for a dataset (first few images)
export const getSampleImages = async (datasetId, limit = 4) => {
    try {
        const response = await fetch(
            `${API_BASE_URL}/datasets/${datasetId}/thumbnails/b64?limit=${limit}`,
            {
                headers: getAuthHeaders(),
            }
        );
        const data = await handleApiError(response);

        if (data.success && data.images) {
            return Object.entries(data.images).map(([id, base64]) => ({
                id: Number(id) || id,
                base64: base64,
                filename: `image_${id}`,
            }));
        }

        return [];
    } catch (error) {
        return [];
    }
};

/**
 * Download a dataset in COCO format for ML tasks.
 *
 * With `includeImages` it downloads a ZIP bundle (COCO JSON + referenced images)
 * from `GET /datasets/{id}/coco`; otherwise it downloads the annotations-only
 * COCO JSON from `GET /datasets/{id}/coco/annotations`. The browser download is
 * triggered automatically.
 *
 * @param {number} datasetId
 * @param {Object} options
 * @param {boolean} [options.includeImages=true] - Bundle images (ZIP) vs annotations-only (JSON).
 * @param {boolean} [options.excludeUnreviewed=true] - Drop contours that haven't been reviewed.
 * @param {boolean} [options.excludeNotFullyAnnotated=true] - Drop images whose masks aren't fully annotated.
 * @param {"all"|"leaves"|"top_level"} [options.contourSelection="all"] - Which contours of the hierarchy to emit.
 */
export const downloadCocoExport = async (
    datasetId,
    {
        includeImages = true,
        excludeUnreviewed = true,
        excludeNotFullyAnnotated = true,
        contourSelection = "all",
    } = {}
) => {
    if (!datasetId) {
        throw new Error("Dataset ID is required");
    }

    const path = includeImages
        ? `/datasets/${datasetId}/coco`
        : `/datasets/${datasetId}/coco/annotations`;

    const params = {
        exclude_unreviewed: excludeUnreviewed,
        exclude_not_fully_annotated: excludeNotFullyAnnotated,
        contour_selection: contourSelection,
    };
    if (includeImages) {
        params.include_images = true;
    }

    const url = buildUrl(API_BASE_URL, path, params);
    const response = await fetch(url, { headers: getAuthHeaders() });

    if (!response.ok) {
        let message = `Export failed (${response.status})`;
        try {
            const data = await response.json();
            message = data.detail || data.message || message;
        } catch (_) {
            // non-JSON error body; keep the status-based message
        }
        throw new Error(message);
    }

    const blob = await response.blob();

    // Prefer the server-provided filename; fall back to a sensible default.
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const fallback = includeImages
        ? `dataset_${datasetId}_coco.zip`
        : `dataset_${datasetId}_coco.json`;
    const filename = match ? match[1] : fallback;

    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(blobUrl);
};