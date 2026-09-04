import { handleApiError, getAuthHeaders, buildUrl } from "../api/util";

import { API_BASE_URL } from "./config";

// Fetch all available labels
export const fetchLabels = async (datasetId) => {
    try {
        if (!datasetId) {
            throw new Error("Dataset ID is required");
        }
        const response = await fetch(
            `${API_BASE_URL}/datasets/${datasetId}/labels`,
            {
                headers: getAuthHeaders(),
            }
        );
        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

// Create a new label (class)
// labelData: { name: string, parent_id: number | null, value: number | null }
// parent_id: null for top-level labels, actual ID for subclasses
export const createLabel = async (labelData, datasetId) => {
    try {
        // Extract values from the label data object
        const { name, parent_id = null, value = null } = labelData;

        if (!name) {
            throw new Error("Label name is required");
        }

        if (!datasetId) {
            throw new Error("Dataset ID is required");
        }

        const urlParams = {
            label_name: name,
            dataset_id: datasetId
        };

        // Send null for top-level, actual ID for subclasses
        if (parent_id !== null) {
            urlParams.parent_label_id = parent_id;
        }

        // Send value if provided
        if (value !== null) {
            urlParams.label_value = value;
        }

        const url = buildUrl(API_BASE_URL, '/labels/create', urlParams);

        const response = await fetch(url, {
            method: "POST",
            headers: getAuthHeaders(),
        });

        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

// Update a label
// labelData: { name: string }
export const updateLabel = async (labelId, labelData, datasetId) => {
    try {
        if (!labelId) {
            throw new Error("Label ID is required");
        }

        if (!labelData || !labelData.name) {
            throw new Error("Label name is required");
        }

        const response = await fetch(
            `${API_BASE_URL}/labels/${labelId}`,
            {
                method: "PATCH",
                headers: {
                    ...getAuthHeaders(),
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ name: labelData.name }),
            }
        );

        return handleApiError(response);
    } catch (error) {
        throw error;
    }
};

// How the objects carrying this label are nested today:
// { nested_total, by_container_label: { <container label id>: count } }.
//
// Fetched once when a drag starts so every row can be priced without a request per
// hover: moving the label under P strands `nested_total - by_container_label[P]`
// objects, and all of them when moving to the top level. Advisory only — the move
// endpoint re-derives it and can still refuse.
export const fetchLabelNestingSummary = async (labelId) => {
    if (!labelId) {
        throw new Error("Label ID is required");
    }

    const response = await fetch(`${API_BASE_URL}/labels/${labelId}/nesting_summary`, {
        headers: getAuthHeaders(),
    });

    return handleApiError(response);
};

// Move a label under a different parent, or to the top level with newParentId = null.
//
// Nesting means part-of, and annotation enforces it, so a move can invalidate objects
// that were legal when they were drawn. The backend refuses such a move with a 409 and
// describes what it would break; that is returned as { blocked: true } rather than
// thrown, because it is a decision for the user, not an error.
//
// Returns { success: true, detached_count } | { blocked: true, message, affectedCount,
// affectedObjects }.
export const moveLabel = async (labelId, newParentId, { detachAffected = false } = {}) => {
    if (!labelId) {
        throw new Error("Label ID is required");
    }

    const response = await fetch(`${API_BASE_URL}/labels/${labelId}/move`, {
        method: "POST",
        headers: {
            ...getAuthHeaders(),
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            new_parent_id: newParentId ?? null,
            detach_affected: detachAffected,
        }),
    });

    if (response.status === 409) {
        const body = await response.json().catch(() => ({}));
        const detail = body?.detail ?? {};
        return {
            blocked: true,
            message: detail.message ?? "This move would invalidate existing annotations.",
            affectedCount: detail.affected_count ?? 0,
            affectedObjects: detail.affected_objects ?? [],
        };
    }

    return handleApiError(response);
};

// Delete a label
export const deleteLabel = async (labelId, datasetId) => {
    try {
        if (!labelId) {
            throw new Error("Label ID is required");
        }

        const response = await fetch(
            `${API_BASE_URL}/labels/${labelId}`,
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